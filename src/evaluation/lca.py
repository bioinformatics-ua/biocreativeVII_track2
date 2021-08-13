import codecs
import collections
import gzip
import json
import logging
import math

import profiler

# Returns precision, recall & f-score for the specified reference and prediction files

log = logging.getLogger(__name__)

class lca_hierarchy:

	def __init__(self, root):
		self.root = root
		self.child2parents = dict()

	def load_parents(self, parents_filename):
		global child2parents
		# Load child2parents
		if parents_filename.endswith(".gz"):
			parents_file = gzip.open(parents_filename, 'rt', encoding="utf-8") 
		else:
			parents_file = codecs.open(parents_filename, 'r', encoding="utf-8") 
		self.child2parents = json.load(parents_file)
		parents_file.close()
		# Verify all parents listed
		children = set(self.child2parents.keys())
		parents = set()
		for parent_list in self.child2parents.values():
			parents.update(parent_list)
		assert len(parents - children) == 0
		del parents
		# Add fake root
		needs_root = {item for item in self.child2parents.keys() if len(self.child2parents[item]) == 0}
		for item in needs_root:
			self.child2parents[item].append(self.root)
		self.child2parents[self.root] = list()

	def prep_ancestor_paths_dict(self, identifiers, ancestor_paths_dict):
		profiler.start()
		for identifier in identifiers:
			self.get_ancestor_paths(identifier, ancestor_paths_dict)
		profiler.stop()

	def get_ancestor_paths(self, identifier, ancestor_paths_dict):
		# Format of ancestor_paths_dict is:
		# ancestor_paths_dict[from_id][ancestor_id] = {set of ids in path}
		log.debug("get_ancestor_paths: {}".format(identifier))
		if identifier in ancestor_paths_dict:
			profiler.count("ancestor_paths_dict:hit")
			return
		profiler.count("ancestor_paths_dict:miss")
		ancestor_paths = dict()
		ancestor_paths[identifier] = {identifier}
		for parent in self.child2parents[identifier]:
			self.get_ancestor_paths(parent, ancestor_paths_dict)
			for ancestor, ancestor_path in ancestor_paths_dict[parent].items():
				ancestor_paths[ancestor] = ancestor_path.union({identifier})
		ancestor_paths_dict[identifier] = ancestor_paths
	
	def get_lca_paths_set(self, identifier1, identifier_set2, ancestor_paths_dict):
		# Returns the set of shortest paths from identifier1 to any identifier in identifier_set2
		profiler.start()
		ancestor_paths1 = ancestor_paths_dict[identifier1]
		ancestors1 = set(ancestor_paths1.keys())
		lca_len = math.inf
		lca_paths = set()
		for identifier2 in identifier_set2:
			ancestor_paths2 = ancestor_paths_dict[identifier2]
			common_ancestors = ancestors1.intersection(ancestor_paths2.keys())
			for ancestor in common_ancestors:
				lca_path = ancestor_paths1[ancestor].union(ancestor_paths2[ancestor])
				lca_path.discard(self.root)
				lca_path_len = len(lca_path)
				if lca_path_len > lca_len:
					continue
				lca_path = list(lca_path)
				lca_path.sort()
				lca_path = tuple(lca_path)
				if lca_path_len < lca_len:
					lca_len = lca_path_len
					lca_paths = {lca_path}
				else:
					lca_paths.add(lca_path)
		profiler.stop()		
		return lca_paths

	def get_all_paths_set(self, identifier_set1, identifier_set2, ancestor_paths_dict):
		profiler.start()		
		paths = set()
		for identifier1 in identifier_set1:
			lca_paths = self.get_lca_paths_set(identifier1, identifier_set2, ancestor_paths_dict)
			paths.update(lca_paths)
		for identifier2 in identifier_set2:
			lca_paths = self.get_lca_paths_set(identifier2, identifier_set1, ancestor_paths_dict)
			paths.update(lca_paths)
		profiler.stop()		
		return paths
	
	def get_minimal_bridges(self, identifier_set1, identifier_set2, paths_available):
		profiler.start()		
		# Identifies the minimal set of additional nodes needed for each identifier in set1 to be connected to at least one identifier in set2, and vica-versa, using greedy approach
		log.debug("Getting minimal bridges for {}, {}".format(identifier_set1, identifier_set2))
		
		# Identify paths required (because it is the only path available)
		identifiers_accepted = set()
		identifier2paths = collections.defaultdict(set)
		for path in paths_available:
			for identifier in identifier_set1.intersection(set(path)):
				identifier2paths[(identifier, 1)].add(path)
			for identifier in identifier_set2.intersection(set(path)):
				identifier2paths[(identifier, 2)].add(path)
		for identifier_tuple, path_set in identifier2paths.items():
			if len(path_set) == 1:
				log.debug("Path {} required for {}".format(path_set, identifier_tuple))
				for path in path_set:
					identifiers_accepted.update(path)
				paths_available -= path_set
		del identifier2paths
		
		iteration = 0
		not_covered1 = identifier_set1 - identifiers_accepted
		not_covered2 = identifier_set2 - identifiers_accepted
		while len(not_covered1) + len(not_covered2) > 0:
			log.debug("Iteration {} len(identifiers_accepted) = {} len(paths_available) = {} not_covered1 = {} not_covered2 = {}".format(iteration, len(identifiers_accepted), len(paths_available), not_covered1, not_covered2))
			# Identify the path that will cover the most for the number of additional identifiers introduced
			best = None
			for index, path in enumerate(paths_available):
				additional_covered1 = identifier_set1.intersection(path) - identifiers_accepted
				additional_covered2 = identifier_set2.intersection(path) - identifiers_accepted
				additional_covered = len(additional_covered1) + len(additional_covered2)
				if additional_covered == 0:
					continue
				additional_cost = len(set(path) - identifiers_accepted - identifier_set1 - identifier_set2)
				path_score = additional_cost / additional_covered
				log.debug("Path #{} = {}, additional_covered {} additional_cost {} score {}".format(index, path, additional_covered, additional_cost, path_score))
				if best is None or path_score < best[0]:
					best = (path_score, path)
			log.debug("Best path = {} score {}".format(best[1], best[0]))
			identifiers_accepted.update(path)
			paths_available -= {path}
			not_covered1 = identifier_set1 - identifiers_accepted
			not_covered2 = identifier_set2 - identifiers_accepted
			iteration += 1

		profiler.stop()		
		return identifiers_accepted
		
	def get_augmented_sets(self, identifier_set1, identifier_set2):
		profiler.start()

		log.debug("Getting augmented sets for {}, {}".format(identifier_set1, identifier_set2))

		# Unknown identifiers cannot be augmented
		removed1 = set(identifier_set1 - self.child2parents.keys())
		removed2 = set(identifier_set2 - self.child2parents.keys())
		log.debug("identifier_set1 unknown identifiers {} identifier_set2 unknown identifiers {}".format(removed1, removed2))
		identifer_set1_internal = identifier_set1 - removed1
		identifer_set2_internal = identifier_set2 - removed2

		if len(identifer_set1_internal) == 0 or len(identifer_set2_internal) == 0:
			log.debug("Known identifier set empty: len(identifer_set1_internal) = {} len(identifer_set1_internal) = {}".format(len(identifer_set1_internal), len(identifer_set2_internal)))
			profiler.stop()		
			profiler.log(log)
			return identifier_set1, identifier_set2

		# TODO Remove identifiers that are ancestors of another identifier
		profiler.start("get_augmented_sets:get_ancestors")
		ancestor_paths_dict = dict()
		self.prep_ancestor_paths_dict(identifer_set1_internal, ancestor_paths_dict)
		self.prep_ancestor_paths_dict(identifer_set2_internal, ancestor_paths_dict)
		ancestors1 = set()
		for identifier in identifer_set1_internal:
			ancestors1.update(ancestor_paths_dict[identifier].keys())
		ancestors2 = set()
		for identifier in identifer_set2_internal:
			ancestors2.update(ancestor_paths_dict[identifier].keys())
		profiler.stop("get_augmented_sets:get_ancestors")

		# Get available paths
		paths_available = self.get_all_paths_set(identifer_set1_internal, identifer_set2_internal, ancestor_paths_dict)
		if log.isEnabledFor(logging.DEBUG):
			for i, path in enumerate(paths_available):
				log.debug("Path {}: {}".format(i, path))

		minimal_bridges = self.get_minimal_bridges(identifer_set1_internal, identifer_set2_internal, paths_available)
		augmented1 = minimal_bridges.intersection(ancestors1)
		augmented1.update(removed1)
		augmented2 = minimal_bridges.intersection(ancestors2)
		augmented2.update(removed2)
		log.debug("len(augmented1) = {} len(augmented2) = {}".format(len(augmented1), len(augmented2)))
		profiler.stop()		
		profiler.log(log)
		return augmented1, augmented2
