import logging
import numpy as np
import os
from graphviz import Digraph
from queue import Queue
import torch
import time
import pickle
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw as Draw
from utils.logger import setup_logger
from utils.prepare_methods import prepare_starting_molecules, prepare_mlp
from utils.smiles_process import reaction_smarts_to_fp, smiles_to_fp
import random
from PIL import Image
from io import BytesIO


class SynRoute:
    def __init__(self, target_mol,):
        self.target_mol = target_mol
        self.mols = [target_mol]
        self.templates = [None]
        self.parents = [-1]
        self.children = [None]
        self.length = 0


    def _add_mol(self, mol, parent_id):
        self.mols.append(mol)

        self.templates.append(None)
        self.parents.append(parent_id)
        self.children.append(None)

        self.children[parent_id].append(len(self.mols)-1)


    def add_reaction(self, mol, template, reactants):
        assert mol in self.mols
        # self.total_cost += cost
        self.length += 1

        parent_id = self.mols.index(mol)

        self.templates[parent_id] = template
        self.children[parent_id] = []
        # self.costs[parent_id] = cost

        for reactant in reactants:
            self._add_mol(reactant, parent_id)

    def viz_route(self, viz_file):
        G = Digraph('G', filename=viz_file)
        G.attr('node', shape='box')
        G.format = 'pdf'

        names = []
        for i in range(len(self.mols)):
            name = self.mols[i]
            # if self.templates[i] is not None:
            #     name += ' | %s' % self.templates[i]
            names.append(name)

        node_queue = Queue()
        node_queue.put((0,-1))   # target mol idx, and parent idx
        while not node_queue.empty():
            idx, parent_idx = node_queue.get()

            if parent_idx >= 0:
                G.edge(names[parent_idx], names[idx], label='')

            if self.children[idx] is not None:
                for c in self.children[idx]:
                    node_queue.put((c, idx))

        G.render()

    def serialize_reaction(self, idx):
        s = self.mols[idx]
        if self.children[idx] is None:
            return s
        # s += '>%.4f>' % np.exp(-self.costs[idx])
        s += '>>'
        s += self.mols[self.children[idx][0]]
        for i in range(1, len(self.children[idx])):
            s += '.'
            s += self.mols[self.children[idx][i]]

        return s

    def serialize(self):
        s = self.serialize_reaction(0)
        for i in range(1, len(self.mols)):
            if self.children[i] is not None:
                s += '|'
                s += self.serialize_reaction(i)

        return s

class MCTS_node:
    def __init__(self, parent, mol_list, reaction_mol, reactant_list, template, prob):
        self.parent = parent
        self.mol_list = mol_list
        self.reaction_mol = reaction_mol
        self.reactant_list = reactant_list
        self.template = template
        self.prob = prob

        self.N_count = 0
        self.Q_value = 0
        self.children = []
        self.success = False
        if parent is not None:
            parent.children.append(self)

    def get_success_child(self):
        for child in self.children:
            if child.success == True:
                return child
        return None

class MolTree:
    def __init__(self, target_mol, known_mols):
        self.target_mol = target_mol

        self.known_mols = known_mols

        self.mol_nodes = []
        self.C = 0.5
        self.root = self._add_mol_node([target_mol], None, None, None, None, None)
        self.root.N_count = 1
        self.succ = target_mol in known_mols
        if self.succ:
            logging.info('Synthesis route found: target in starting molecules')

    def _add_mol_node(self, mol_list, parent, reaction_mol, reactant_list, template, prob):
        # (self, parent, mol_list, reaction_mol, template, prob):
        mol_node = MCTS_node(
            mol_list=mol_list,
            parent = parent,
            reaction_mol = reaction_mol,
            reactant_list = reactant_list,
            template = template,
            prob = prob
        )
        self.mol_nodes.append(mol_node)
        mol_node.id = len(self.mol_nodes)
        return mol_node

    def expand(self, mol_node, reaction_mol, reactant_lists, pros, templates):
        assert len(mol_node.children) == 0

        if pros is None:
            return None
        else :

            max_prob_node = None
            max_prob = max(pros)

            for i in range(len(pros)):

                assert pros[i] > 0
                new_mol_list = mol_node.mol_list[:]
                new_mol_list.remove(reaction_mol)
                new_mol_list.extend(reactant_lists[i])

                succ = True
                for mol in new_mol_list:
                    if mol not in self.known_mols:
                        succ = False
                        break
                node = self._add_mol_node(mol_list = new_mol_list, parent = mol_node, reaction_mol = reaction_mol, reactant_list = reactant_lists[i], template = templates[i], prob = pros[i])
                if succ is True:
                    max_prob_node = node
                elif pros[i] == max_prob and max_prob_node is None:
                    max_prob_node = node
            if len(mol_node.children) == 0:
                return None
            return max_prob_node

    def get_best_route(self):
        if not self.succ:
            return None

        syn_route = SynRoute(
            target_mol=self.root.mol_list[0],
        )

        mol_node = self.root
        while 1:
            assert mol_node.success == True
            if len(mol_node.children) == 0:
                break

            mol_node = mol_node.get_success_child()
            assert mol_node is not None
            # print('add reaction:')
            # print('reaction mol:', mol_node.reaction_mol)
            # print('reactants:', mol_node.reactant_list)
            syn_route.add_reaction(
                mol=mol_node.reaction_mol,
                template=mol_node.template,
                reactants=mol_node.reactant_list
            )
        return syn_route



    def update(self, mol_node, new_reward):

        current_mol_node = mol_node
        while 1:

            if current_mol_node.N_count == 0:
                current_mol_node.N_count += 1
                current_mol_node.Q_value = new_reward

            else:
                current_mol_node.Q_value = (current_mol_node.Q_value * current_mol_node.N_count + new_reward)
                current_mol_node.N_count += 1
                current_mol_node.Q_value = current_mol_node.Q_value / current_mol_node.N_count
            if current_mol_node == self.root:
                break
            else:
                if current_mol_node.success == True:
                    current_mol_node.parent.success = True
                current_mol_node = current_mol_node.parent
        if current_mol_node.success:
            self.succ = True
            return True
        else:
            self.succ = False
            # print('update end, find fail ')
            return False

def MCTS_rollout(target_mol, target_mol_id, starting_mols, expand_fn, iterations, viz=False, viz_dir=None, rollout_max_depth = 5):
    mol_tree = MolTree(
        target_mol=target_mol,
        known_mols=starting_mols,

    )

    def rollout(mol_node):
        reward = None
        mol_set = mol_node.mol_list[:]
        select_mols = []
        for mol in mol_set:
            if mol not in starting_mols:
                select_mols.append(mol)
        if len(select_mols) == 0:
            reward = 10
            mol_node.success = True
            return reward, 0

        depth = 0
        mol_set = mol_node.mol_list[:]
        while depth < rollout_max_depth:

            select_mols = []
            for mol in mol_set:
                if mol not in starting_mols:
                    select_mols.append(mol)

            if len(select_mols) == 0:
                reward = 1
                break
            select_index = np.random.randint(0, len(select_mols))

            rollout_mol = select_mols[select_index]
            rollout_result = expand_fn(rollout_mol)

            if rollout_result is not None and (len(rollout_result['scores']) > 0):
                lens = 10 if len(rollout_result['reactants']) >= 10 else len(rollout_result['reactants'])
                rollout_reactants = rollout_result['reactants'][0:lens]
                rollout_pros = rollout_result['scores'][0:lens]
                rollout_reactant_lists = []
                for j in range(len(rollout_pros)):
                    rollout_reactant_list = list(set(rollout_reactants[j].split('.')))
                    rollout_reactant_lists.append(rollout_reactant_list)

                select_reaction_index = np.random.randint(0, len(rollout_pros))
                mol_set.remove(rollout_mol)
                mol_set.extend(rollout_reactant_lists[select_reaction_index])
            depth += 1

        if reward is None:
            in_number = 0
            for mol in mol_set:
                if mol in starting_mols:
                    in_number += 1
            reward = (in_number / len(mol_set))
        # print('use rollout get score ', reward, ' depth ',depth)
        return reward, depth
    i = -1
    C = 0.5
    model_call = 0
    t0 = time.time()
    if not mol_tree.succ:
        for i in range(iterations):
            next_node = mol_tree.root
            parent_N_count = next_node.N_count
            while len(next_node.children) != 0:
                Q_values = [mol_node.Q_value for mol_node in next_node.children]
                probs = [mol_node.prob for mol_node in next_node.children]
                N_counts = [mol_node.N_count for mol_node in next_node.children]
                a_values = [Q_values[i]/N_counts[i] + C*probs[i]*((parent_N_count)**0.5)/(1+N_counts[i]) if N_counts[i]!=0 else C*probs[i]*((parent_N_count)**0.5)/(1+N_counts[i]) for i in range(len(Q_values))]
                next_node = next_node.children[np.argmax(a_values)]
            if next_node.N_count == 0 and next_node != mol_tree.root:
                reward, rollout_depth= rollout(next_node)
                model_call += rollout_depth
                succ = mol_tree.update(next_node, reward)
                if succ:
                    break

            else:
                assert next_node.success == False
                mol_set = next_node.mol_list[:]
                select_mols = []
                for mol in mol_set:
                    if mol not in starting_mols:
                        select_mols.append(mol)
                if len(select_mols)==0:
                    print(next_node.N_count, next_node.Q_value)
                select_index = np.random.randint(0, len(select_mols))

                expand_mol = select_mols[select_index]
                result = expand_fn(expand_mol)
                model_call += 1

                if result is not None and (len(result['scores']) > 0):
                    # print('iter ',iterations,' expand on ',expand_mol, ' success ',len(result['scores']))
                    reactants = result['reactants']
                    pros = result['scores']
                    if 'templates' in result.keys():
                        templates = result['templates']
                    else:
                        templates = result['template']
                    reactant_lists = []
                    for j in range(len(pros)):
                        reactant_list = list(set(reactants[j].split('.')))
                        reactant_lists.append(reactant_list)
                    # expand(self, mol_node, reaction_mol, reactant_lists, pros, templates):
                    max_prob_node = mol_tree.expand(next_node, expand_mol, reactant_lists, pros, templates)
                    if max_prob_node is None:

                        reward = -10
                        succ = mol_tree.update(next_node, reward)
                    else:
                        reward, rollout_depth = rollout(max_prob_node)
                        model_call += rollout_depth
                        succ = mol_tree.update(max_prob_node, reward)
                        if succ:
                            break
                else:
                    # print('iter ',iterations,' expand on ', expand_mol, ' failed ')
                    mol_tree.expand(next_node, None, None, None, None)
                    reward = -10
                    succ = mol_tree.update(next_node, reward)
            if succ:
                break
        logging.info('information: reation_node number:  %d  |  mol_node number:  %d  | iter:  %d' % (
        0, len(mol_tree.mol_nodes), i+1))
        logging.info('Final search status | expand model call | value model call | iter: %s | %d | %d | %d'
                     % (str(mol_tree.succ), model_call, 0, i+1))

    best_route = None
    if mol_tree.succ:
        best_route = mol_tree.get_best_route()
        assert best_route is not None

    return mol_tree.succ, best_route, (i + 1, model_call, 0, 0, len(mol_tree.mol_nodes))
    # return False, None, (self.count, self.expand_model_call, self.value_model_call, len(self.reaction_nodes), len(self.mol_nodes))

def prepare_MCTS_rollout_planner(one_step, starting_mols, expansion_topk, iterations):
    expansion_handle = lambda x: one_step.run(x, topk=expansion_topk)
    plan_handle =  lambda x, y=0: MCTS_rollout(
        target_mol=x,
        target_mol_id=y,
        starting_mols=starting_mols,
        expand_fn=expansion_handle,
        iterations=iterations,
    )

    return plan_handle


def MCTS_rollout_plan(test_file_name, iterations, result_file_name):


    starting_mols = prepare_starting_molecules('dataset/origin_dict.csv')
    routes = []

    for line in open(test_file_name, "r"):
        routes.append(line.strip())
    logging.info('%d routes extracted from %s loaded' % (len(routes),
                                                         test_file_name))
    mlp_templates = 'one_step_model/template_rules_1.dat'
    mlp_model_dump = 'one_step_model/retro_star_value_ours.ckpt'
    one_step = prepare_mlp(mlp_templates, mlp_model_dump)

    plan_handle = prepare_MCTS_rollout_planner(
        one_step=one_step,
        starting_mols=starting_mols,
        expansion_topk=50,
        iterations=iterations,

    )

    result = {
        'succ': [],
        'cumulated_time': [],
        'iter': [],
        'routes': [],
        'route_lens': [],
        'reaction_nodes_lens': [],
        'mol_nodes_lens': [],
        'expand_model_call': [],
        'value_model_call': []
    }
    num_targets = len(routes)
    t0 = time.time()



    for (i, route) in enumerate(routes):
        target_mol = route
        print(i, target_mol)

        try:

            # (self.count, self.expand_model_call, self.value_model_call, len(self.reaction_nodes), len(self.mol_nodes))
            succ, route, msg = plan_handle(target_mol, i)

            result['succ'].append(succ)
            result['cumulated_time'].append(time.time() - t0)
            result['iter'].append(msg[0])
            result['routes'].append(route)
            result['expand_model_call'].append(msg[1])
            result['value_model_call'].append(msg[2])
            result['reaction_nodes_lens'].append(msg[3])
            result['mol_nodes_lens'].append(msg[4])
            if succ:
                # result['route_costs'].append(msg[0].total_cost)
                result['route_lens'].append(route.length)
            else:
                # result['route_costs'].append(None)
                result['route_lens'].append(None)

            tot_num = i + 1

            tot_succ = np.array(result['succ']).sum()
            avg_time = (time.time() - t0) * 1.0 / tot_num
            avg_expand_model_call = np.array(result['expand_model_call'], dtype=float).mean()
            avg_value_model_call = np.array(result['value_model_call'], dtype=float).mean()
            avg_iter = np.array(result['iter'], dtype=float).mean()
            avg_reaction_nodes_number = np.array(result['reaction_nodes_lens'], dtype=float).mean()
            avg_mol_nodes_number = np.array(result['mol_nodes_lens'], dtype=float).mean()
            logging.info('Succ: %d/%d/%d | avg time: %.2f s | avg iter: %.2f | avg expand_model_call: %.2f | avg value_model_call: %.2f | avg reaction_nodes_number: %.2f | avg mol_nodes_number: %.2f' %
                         (tot_succ, tot_num, num_targets, avg_time, avg_iter, avg_expand_model_call, avg_value_model_call, avg_reaction_nodes_number, avg_mol_nodes_number))
        except :
            logging.info(' failed')
            result['succ'].append(False)
            result['cumulated_time'].append(time.time() - t0)
            result['iter'].append(500)
            result['routes'].append(None)
            result['route_lens'].append(None)
            result['expand_model_call'].append(msg[1])
            result['value_model_call'].append(msg[2])
            result['reaction_nodes_lens'].append(msg[3])
            result['mol_nodes_lens'].append(msg[4])


    f = open('MCTS_rollout_results_' + result_file_name, 'wb')
    pickle.dump(result, f)
    f.close()

if __name__ == '__main__':
    np.random.seed(1234)
    torch.manual_seed(1234)
    random.seed(1234)

    setup_logger('MCTS_rollout_plan_retro190_log.log')
    MCTS_rollout_plan('dataset/retro190.txt', 500, 'MCTS_rollout_plan_retro190_results')