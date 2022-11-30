import torch

import pickle

from utils.prepare_methods import prepare_starting_molecules, prepare_mlp
from utils.logger import setup_logger

from rdkit import Chem
import os


import numpy as np
import logging
import time
class DFS_node:

    def __init__(self, mols, template, prob , parent , reaction_mol , reactants ):
        self.mols = mols
        self.parent = parent
        self.reaction_mol = reaction_mol
        self.children = []
        self.prob = prob
        self.template = template
        self.reactants = reactants
        if parent is not None:
            self.parent.children.append(self)



class DFS_tree:
    def __init__(self,known_mols, expand_fn, max_depth, iterations):

        self.reaction_nodes_number = 0
        self.value_model_call = 0
        self.known_mols = known_mols
        self.expand_fn = expand_fn
        self.max_depth = max_depth
        self.search_fail = []
        self.iterations = iterations

        self.count = 0
        self.mol_nodes_number = 0
        self.expand_model_call = 0
        self.ancestors = []

    def DFS(self, node, depth):

        mols = node.mols

        succe = True
        un_succ_mol = []
        for mol in mols:
            if mol not in self.known_mols:
                succe = False
                un_succ_mol.append(mol)

        if succe:

            path = [node]

            return path
        if self.count >= self.iterations:
            return None

        if depth >= self.max_depth:
            # self.count += 1
            return None

        select_index = np.random.randint(0,len(un_succ_mol))
        select_mol = un_succ_mol[select_index]
        # print('iteration %d expand on mol %s'%(self.count, select_mol))
        result = self.expand_fn(select_mol)
        self.expand_model_call += 1
        self.count += 1

        if result is not None and (len(result['scores']) > 0):
            reactants = result['reactants']
            pros = result['scores']
            if 'templates' in result.keys():
                templates = result['templates']
            else:
                templates = result['template']

            reactant_lists = []
            pros_sum = 0
            for j in range(len(pros)):
                pros_sum += pros[j]
                reactant_list = list(set(reactants[j].split('.')))
                reactant_lists.append(reactant_list)
                if pros_sum > 0.995:
                    break

            for i in range(len(reactant_lists)):

                new_node_mols = mols[:]
                repeat_mol = False

                for mol in reactant_lists[i]:
                    if mol in new_node_mols :
                        repeat_mol = True
                        break
                if repeat_mol:
                    continue
                new_node_mols.remove(select_mol)
                if new_node_mols is None:
                    new_node_mols = []


                new_node_mols.extend(reactant_lists[i])
                # print('add reactants ',new_node_mols)
                fail_node = False
                for mol in new_node_mols:
                    if mol in self.search_fail:
                        fail_node = True
                        break
                if fail_node:
                    continue

                new_node = DFS_node(new_node_mols, templates[i], prob=pros[i], parent=node, reaction_mol=select_mol,reactants = reactant_lists[i])
                self.mol_nodes_number += 1
                new_depth = depth + 1
                #  answer= None
                answer = self.DFS(new_node, new_depth)
                # logging.info('answer %s'%answer)
                if answer is not None:
                    answer.append(node)
                    return answer
                else :
                    continue
            return None
        else:
            self.search_fail.append(select_mol)

            return None



    def search(self,target_mol):

        target_mol = target_mol.replace("\n","")
        root = DFS_node([target_mol],None,0,None,None,None)

        self.count = 0
        self.mol_nodes_number = 1
        self.expand_model_call = 0
        self.ancestors.clear()


        try:
            path = self.DFS(root, 0)
            if path is None:
                return None, (self.count, self.expand_model_call, self.value_model_call, self.mol_nodes_number, self.reaction_nodes_number)
            path.reverse()
            return path, (self.count, self.expand_model_call, self.value_model_call, self.mol_nodes_number, self.reaction_nodes_number)
        except:
            return None, (self.count, self.expand_model_call, self.value_model_call, self.mol_nodes_number, self.reaction_nodes_number)



def prepare_DFS_planner(one_step, starting_mols, expansion_topk, max_depth,iterations):
    expansion_handle = lambda x: one_step.run(x, topk=expansion_topk)
    plan_handle = DFS_tree(
        starting_mols,
        expansion_handle,
        max_depth,
        iterations
    )
    # print('iterations',iterations)
    return plan_handle

class RSPlanner:
    def __init__(self,
                 gpu=-1,
                 expansion_topk=50,
                 starting_molecules='dataset/origin_dict.csv', # origin_dict
                 mlp_templates='one_step_model/template_rules_1.dat',
                 mlp_model_dump='one_step_model/retro_star_value_ours.ckpt',
                 max_depth=15,
                 iterations=500
                 ):

        device = torch.device('cuda:%d' % gpu if gpu >= 0 else 'cpu')
        starting_mols = prepare_starting_molecules(starting_molecules)

        one_step = prepare_mlp(mlp_templates, mlp_model_dump)

        self.plan_handle = prepare_DFS_planner(
            one_step=one_step,
            starting_mols=starting_mols,
            expansion_topk=expansion_topk,
            max_depth=max_depth,
            iterations = iterations

        )

    def plan(self, target_mol):

        # (self.count, self.expand_model_call, self.value_model_call, self.mol_nodes_number, self.reaction_nodes_number)
        return self.plan_handle.search(target_mol)




def DFS_plan(test_file_name, iterations, result_file_name):
    routes = []

    for line in open(test_file_name, "r"):
        routes.append(line.strip())
    logging.info('%d routes extracted from %s loaded' % (len(routes),
                                                         test_file_name))

    planner = RSPlanner(
        gpu=-1,
        expansion_topk=50,
        max_depth=10,
        iterations=iterations

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
        msg = None
        try:
            # (self.count, self.expand_model_call, self.value_model_call, self.mol_nodes_number, self.reaction_nodes_number)
            path, msg = planner.plan(target_mol)
            path_list = None
            if path is not None :
                result['succ'].append(1)
                result['route_lens'].append(len(path))
                path_list = []
                for node in path:
                    if node.template is None:
                        continue
                    # self, mols, template, prob , parent , reaction_mol , reactants ):
                    reaction = ''
                    reaction += node.reaction_mol
                    reaction = reaction + '>>'
                    for reactant in node.reactants:
                        reaction = reaction + reactant + '.'
                    path_list.append(reaction)
                    # print(path_list)
                logging.info('information: reation_node number:  %d  |  mol_node number:  %d  | iter:  %d' % (
                msg[3], msg[4], msg[0]))
                logging.info('Final search status | expand model call | value model call | iter: %s | %d | %d | %d'
                                 % ('True', msg[1], msg[2], msg[0]))
            else :
                result['succ'].append(0)
                result['route_lens'].append(None)
                logging.info('information: reation_node number:  %d  |  mol_node number:  %d  | iter:  %d' % (
                    msg[3], msg[4], msg[0]))
                logging.info('Final search status | expand model call | value model call | iter: %s | %d | %d | %d'
                             % ('False', msg[1], msg[2], msg[0]))

            result['cumulated_time'].append(time.time() - t0)
            result['iter'].append(msg[0])
            result['routes'].append(path_list)
            result['expand_model_call'].append(msg[1])
            result['value_model_call'].append(msg[2])
            result['reaction_nodes_lens'].append(msg[3])
            result['mol_nodes_lens'].append(msg[4])
            tot_num = i + 1

            tot_succ = np.array(result['succ']).sum()
            avg_time = (time.time() - t0) * 1.0 / tot_num
            avg_expand_model_call = np.array(result['expand_model_call'], dtype=float).mean()
            avg_value_model_call = np.array(result['value_model_call'], dtype=float).mean()
            avg_iter = np.array(result['iter'], dtype=float).mean()
            avg_reaction_nodes_number = np.array(result['reaction_nodes_lens'], dtype=float).mean()
            avg_mol_nodes_number = np.array(result['mol_nodes_lens'], dtype=float).mean()

            logging.info(
                'Succ: %d/%d/%d | avg time: %.2f s | avg iter: %.2f | avg expand_model_call: %.2f | avg value_model_call: %.2f | avg reaction_nodes_number: %.2f | avg mol_nodes_number: %.2f' %
                (tot_succ, tot_num, num_targets, avg_time, avg_iter, avg_expand_model_call, avg_value_model_call,
                 avg_reaction_nodes_number, avg_mol_nodes_number))
        except:
            logging.info(' failed')
            result['succ'].append(False)
            result['cumulated_time'].append(None)
            result['iter'].append(500)
            result['routes'].append(None)
            result['route_lens'].append(None)
            result['expand_model_call'].append(msg[1])
            result['value_model_call'].append(msg[2])
            result['reaction_nodes_lens'].append(msg[3])
            result['mol_nodes_lens'].append(msg[4])

    f = open('DFS_results_' + result_file_name, 'wb')
    pickle.dump(result, f)
    f.close()


if __name__ == '__main__':


    setup_logger('DFS_plan_retro190_log.log')
    DFS_plan('dataset/retro190.txt', 500, 'DFS_plan_retro190_results')






