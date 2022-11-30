import numpy as np
import logging
import math
import numpy as np
from queue import Queue
from graphviz import Digraph
import torch
import random
import logging
import time
import pickle
import os
from rdkit import Chem, DataStructs
from utils.logger import setup_logger
from utils.prepare_methods import prepare_starting_molecules, prepare_mlp
from utils.smiles_process import reaction_smarts_to_fp, smiles_to_fp
import random
from PIL import Image
from io import BytesIO
from PIL import Image
from io import BytesIO
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw as Draw


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
# or-node
class MolNode:
    def __init__(self, mol , parent):
        self.mol = mol
        self.id = 0
        self.parent = parent
        self.children = []
        self.pn = 1
        # dn disprove number if dn = 0 or pn = NINF fail
        self.dn = 1
        # random
        self.thpn = 100000
        self.thdn = 100000
        self.children_expand = 0
        if parent is not None:
            parent.children.append(self)

    def set_disprove(self):
        self.pn = np.Inf
        self.dn = 0

    def set_prove(self):
        self.pn = 0
        self.dn = np.Inf
    def all_child_expand(self):
        return self.children_expand == len(self.children)

    def select_sbest_and_s2(self):

        pn_list = [child_node.pn + child_node.h_value for child_node in self.children]
        sbest_index = np.argmin(pn_list)
        sbest = self.children[sbest_index]

        min_pn = np.min(pn_list)
        pn_list2 = [pn_list[i] if pn_list[i]!=min_pn else 1000000 for i in range(len(pn_list))]
        s2_index = np.argmin(pn_list2)
        s2 = self.children[s2_index]
        return sbest, s2

    def select_best_reaction(self):
        pn_list = [child_node.pn  for child_node in self.children]
        sbest_index = np.argmin(pn_list)
        sbest = self.children[sbest_index]
        return  sbest

    def update(self):
        dn_list = [child_node.dn for child_node in self.children]
        new_dn = sum(dn_list)
        if new_dn == np.inf:
            self.dn = new_dn
            self.pn = 0
            # logging.info('success on mol_node %d'%(self.id))
        else:
            self.dn = new_dn
            # pn + h_value
            pn_list = [child_node.pn + child_node.h_value for child_node in self.children]
            self.pn = np.min(pn_list)

# and-node
class ReactionNode:
    def __init__(self, template, reactants_list, parent, h_value):
        self.template = template
        self.reactants_list = reactants_list
        self.parent = parent
        self.children = []
        self.id = 0
        self.h_value = h_value
        self.pn = 1
        self.dn = 1
        self.thpn = 100000
        self.thdn = 100000
        parent.children.append(self)

    def select_sbest_and_s2(self):

        dn_list = [child_node.dn for child_node in self.children]
        min_dn = np.argmin(dn_list)
        # min_list = [1 if dn_list[i] == min_pn else 0 for i in range(len(dn_list))]
        # sbest_indexs = np.where(np.array(min_list)==1)
        sbest_index = np.argmin(dn_list)
        sbest = self.children[sbest_index]

        dn_list2 = [dn_list[i] if dn_list[i]!=min_dn else 1000000 for i in range(len(dn_list))]
        s2_index = np.argmin(dn_list2)
        s2 = self.children[s2_index]
        return sbest, s2

    def update(self):
        pn_list = [child_node.pn for child_node in self.children]
        new_pn = sum(pn_list)
        self.pn = new_pn
        dn_list = [child_node.dn for child_node in self.children]
        self.dn = np.min(dn_list)



class MolTree:
    def __init__(self, starting_mols, target_mol, target_mol_id, expand_fn, iterations):
        self.staring_mols = starting_mols
        self.expand_fn = expand_fn
        self.target_mol = target_mol
        self.target_mol_id = target_mol_id
        self.iteration = iterations
        self.mol_nodes = []

        self.reaction_nodes = []
        self.root = self._add_mol_node(target_mol, None)
        self.M_pn = 20
        self.controll = 1e-30
        self.paramenter = 2
        self.count = 0
        self.expand_model_call = 0
        self.value_model_call = 0


    def _add_mol_node(self, mol, parent):
        mol_node = MolNode(
            mol = mol,
            parent = parent
        )
        self.mol_nodes.append(mol_node)
        mol_node.id = len(self.mol_nodes)
        return mol_node

    # expand
    def _add_reaction_node(self, template, reactants_list, parent ,h_value):
        reaction_node = ReactionNode(template, reactants_list, parent, h_value)
        self.reaction_nodes.append(reaction_node)
        reaction_node.id = len(self.reaction_nodes)
        return reaction_node

    def mol_node_expand(self, mol_node):
        result = self.expand_fn(mol_node.mol)
        self.expand_model_call += 1
        self.count += 1
        if result is not None and (len(result['scores']) > 0):
            if 'templates' in result.keys():
                templates = result['templates']
            else:
                templates = result['template']
            reactants = result['reactants']
            pros = result['scores']
            # print('iteration %d expand on mol node %d child_len %d' % (self.count, mol_node.id, len(pros)))

            h_pros = [math.floor(-math.log(pro + self.controll) + 1) for pro in pros]
            h_values = [self.M_pn if h_pro > self.M_pn else h_pro for h_pro in h_pros]


            for j in range(len(pros)):
                reactants_list = list(set(reactants[j].split('.')))
                reaction_node = self._add_reaction_node(templates[j], reactants_list, mol_node, h_values[j])
                self.reaction_node_expand(reaction_node)
                reaction_node.update()
            return True

        else:

            return False

    def reaction_node_expand(self, reaction_node):

        for reactant in reaction_node.reactants_list:
            mol_node = self._add_mol_node(reactant, reaction_node)
            if mol_node.mol in self.staring_mols:
                mol_node.set_prove()

    # mol node =  or node
    def search_on_mol_node(self, mol_node):

        if mol_node.mol in self.staring_mols:
            mol_node.set_prove()
            # logging.info('mol node %d in bb' % (mol_node.id))
            return

        if len(mol_node.children) == 0:
            succ = self.mol_node_expand(mol_node)
            if succ is False:
                mol_node.set_disprove()
                # logging.info('expand fail on mol node %d ' % (mol_node.id))
                return

        while 1:
            if self.count >= self.iteration:
                break

            # logging.info('update on mol node %d, pn %f, dn %f, thpn %f, thdn %f' % (mol_node.id, mol_node.pn, mol_node.dn, mol_node.thpn, mol_node.thdn ))
            mol_node.update()
            # logging.info('update on mol node %d, new pn %f, new dn %f ' % (mol_node.id, mol_node.pn, mol_node.dn))

            if mol_node.thpn <= mol_node.pn or mol_node.thdn <= mol_node.dn:
                # logging.info('th chaochu on mol node %d'% (mol_node.id))
                break

            sbest, s2 = mol_node.select_sbest_and_s2()
            # update sbest thpn and thdn
            # logging.info('update on rea node %d, pn %f, dn %f, thpn %f, thdn %f' % (sbest.id, sbest.pn, sbest.dn, sbest.thpn, sbest.thdn))
            # logging.info('s2.pn %s ,sbest.h_value %s'%(s2.pn, sbest.h_value))
            sbest.thpn = min(mol_node.thpn, s2.pn + s2.h_value +self.paramenter) - sbest.h_value
            # sbest.thpn = min(mol_node.thpn, s2.pn + 1)
            sbest.thdn = mol_node.thdn - mol_node.dn + sbest.dn
            # logging.info('update on rea node %d, pn %f, dn %f, thpn %f, thdn %f' % (sbest.id, sbest.pn, sbest.dn, sbest.thpn, sbest.thdn))
            self.search_on_reaction_node(sbest)

    # reaction_node and
    def search_on_reaction_node(self, reaction_node):
        # logging.info('search on rea node %d ' % (reaction_node.id))
        if len(reaction_node.children) == 0:
            self.reaction_node_expand(reaction_node)
        while 1:
            if self.count >= self.iteration:
                break
            # logging.info('update on rea node %d, pn %f, dn %f, thpn %f, thdn %f' % (reaction_node.id, reaction_node.pn, reaction_node.dn, reaction_node.thpn, reaction_node.thdn))
            reaction_node.update()
            # logging.info('update on rea node %d, pn %f, dn %f, thpn %f, thdn %f' % (reaction_node.id, reaction_node.pn, reaction_node.dn, reaction_node.thpn, reaction_node.thdn))
            if reaction_node.thpn <= reaction_node.pn or reaction_node.thdn <= reaction_node.dn:
                # logging.info('th chaochu on rea node %d' % (reaction_node.id))
                break
            sbest, s2 = reaction_node.select_sbest_and_s2()
            # logging.info('update on mol node %d, pn %f, dn %f, thpn %f, thdn %f' % ( sbest.id, sbest.pn, sbest.dn, sbest.thpn, sbest.thdn))
            sbest.thdn = min(reaction_node.thdn, s2.dn + 1)
            sbest.thpn = reaction_node.thpn - reaction_node.pn + sbest.pn
            # logging.info('update on mol node %d, pn %f, dn %f, thpn %f, thdn %f' % (sbest.id, sbest.pn, sbest.dn, sbest.thpn, sbest.thdn))
            self.search_on_mol_node(sbest)


    def search(self):
        t0 = time.time()

        self.search_on_mol_node(self.root)
        if self.root.pn == 0:

            logging.info('information: reation_node number:  %d  |  mol_node number:  %d  | iter:  %d' % (len(self.reaction_nodes),len(self.mol_nodes),self.count))
            logging.info('Final search status | expand model call | value model call | iter: %s | %d | %d | %d'
                         % ('True', self.expand_model_call,self.value_model_call, self.count))

            syn_route = SynRoute(
                target_mol=self.root.mol
            )
            mol_queue = Queue()
            mol_queue.put(self.root)
            while not mol_queue.empty():
                mol = mol_queue.get()
                if len(mol.children) == 0:
                    continue
                assert mol.pn == 0
                best_reaction = mol.select_best_reaction()
                # logging.info('best_reaction.id %d  pn %f dn %f'%(best_reaction.id, best_reaction.pn, best_reaction.dn))
                assert best_reaction.pn == 0

                reactants = []
                for reactant in best_reaction.children:
                    mol_queue.put(reactant)
                    reactants.append(reactant.mol)

                syn_route.add_reaction(
                    mol=mol.mol,
                    template=best_reaction.template,
                    reactants=reactants
                )

            return True, syn_route, (self.count,self.expand_model_call, self.value_model_call, len(self.reaction_nodes), len(self.mol_nodes))
        else:
            logging.info('information: reation_node number:  %d  |  mol_node number:  %d  | iter:  %d' % (
            len(self.reaction_nodes), len(self.mol_nodes), self.count))
            logging.info('Final search status | model call | iter: %s | %d | %d'
                         % ('False', self.count, self.count))
            return False, None, (self.count, self.expand_model_call, self.value_model_call, len(self.reaction_nodes), len(self.mol_nodes))

# def __init__(self, starting_mols, target_mol, expand_fn, iterations):
def DFPN_E(target_mol, target_mol_id, expand_fn, starting_mols, iterations):
    mol_tree = MolTree(
        starting_mols=starting_mols,
        target_mol=target_mol,
        target_mol_id = target_mol_id,
        expand_fn=expand_fn,
        iterations=iterations,

    )
    return mol_tree.search()


def prepare_DFPN_E_planner(one_step, starting_mols, expansion_topk, iterations):
    expansion_handle = lambda x: one_step.run(x, topk=expansion_topk)
    plan_handle = lambda x, y=0: DFPN_E(
        target_mol=x,
        target_mol_id=y,
        starting_mols=starting_mols,
        expand_fn=expansion_handle,
        iterations=iterations,

    )

    return plan_handle


def DFPN_E_plan(test_file_name, iterations, result_file_name):

    starting_mols = prepare_starting_molecules('dataset/origin_dict.csv')
    routes = []

    for line in open(test_file_name, "r"):
        routes.append(line.strip())
    logging.info('%d routes extracted from %s loaded' % (len(routes),
                                                         test_file_name))
    mlp_templates = 'one_step_model/template_rules_1.dat'
    mlp_model_dump = 'one_step_model/retro_star_value_ours.ckpt'
    one_step = prepare_mlp(mlp_templates, mlp_model_dump)


    plan_handle = prepare_DFPN_E_planner(
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

        try :

            # (self.count, self.expand_model_call, self.value_model_call, len(self.reaction_nodes), len(self.mol_nodes))
            succ, route, msg = plan_handle(target_mol)

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


    f = open('DFPN_E_results_' + result_file_name, 'wb')
    pickle.dump(result, f)
    f.close()



if __name__ == '__main__':
    np.random.seed(1234)
    torch.manual_seed(1234)
    random.seed(1234)

    setup_logger('DFPN_E_plan_retro190_log.log')
    DFPN_E_plan('dataset/retro190.txt', 500, 'DFPN_E_plan_retro190_results')


