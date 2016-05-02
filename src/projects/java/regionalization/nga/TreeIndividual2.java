package regionalization.nga;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import spawnn.utils.GraphUtils;

public class TreeIndividual2 extends TreeIndividual {
	
	public TreeIndividual2(Map<double[], Set<double[]>> cm, Map<double[], Set<double[]>> tree, int numCluster) {
		super(cm,tree,numCluster);
	}

	public TreeIndividual2(Map<double[], Set<double[]>> cm, Map<double[], Set<double[]>> tree, Map<double[], Set<double[]>> cuts) {
		super(cm, tree, cuts);
	}
	
	enum NodeState {None,Both,A,B};

	@Override
	public TreeIndividual recombine(TreeIndividual mother) {

		Map<double[], Set<double[]>> treeA = tree;
		Map<double[], Set<double[]>> treeB = mother.getTree();

		Map<double[], Set<double[]>> nTree = new HashMap<double[], Set<double[]>>();
		Map<double[],NodeState> ns = new HashMap<double[],NodeState>();
		for (double[] a : treeA.keySet()) {// init with all nodes
			nTree.put(a, new HashSet<double[]>());
			ns.put(a,NodeState.None);
		}

		for (double[] a : treeA.keySet()) // add intersections
			for (double[] b : treeA.get(a))
				if (treeB.containsKey(a) && treeB.get(a).contains(b)) {
					if (!nTree.containsKey(a))
						nTree.put(a, new HashSet<double[]>());
					nTree.get(a).add(b);
					ns.put(a, NodeState.Both);
					ns.put(b, NodeState.Both);
				}

		List<Map<double[], Set<double[]>>> sub = GraphUtils.getSubGraphs(nTree);
		List<Set<double[]>> added = new ArrayList<Set<double[]>>();
		for (Map<double[], Set<double[]>> m : sub)
			added.add(GraphUtils.getNodes(m));
		
		while (true) {
			// get all candidates with cost for each sub
			Map<Integer,Map<double[], Map<double[],Double>>> c = new HashMap<Integer,Map<double[], Map<double[],Double>>>();
			for (int i = 0; i < sub.size(); i++) {
				Map<double[], Map<double[],Double>> cm = new HashMap<double[], Map<double[],Double>>();
				for (double[] a : added.get(i)) {
										
					Map<double[],Double> s = new HashMap<double[],Double>();
					for (double[] b : treeA.get(a))
						if (!added.get(i).contains(b)) // avoid cycles
							s.put(b, ns.get(a) == NodeState.B ? 10 + r.nextDouble() : r.nextDouble() );
					
					for (double[] b : treeB.get(a))
						if (!added.get(i).contains(b)) // avoid cycles
							s.put(b, ns.get(a) == NodeState.A ? 10 + r.nextDouble() : r.nextDouble() );
					
					if (!s.isEmpty())
						cm.put(a, s);
				}
				if( !cm.isEmpty() )
					c.put(i,cm);
			}
			
			if( c.isEmpty() ) // full?
				break;
			
			// get best candidates
			int idx = -1;
			double[] na = null, nb = null;
			double bestCost = Double.MAX_VALUE;
			for( int i : c.keySet() )
				for( double[] a : c.get(i).keySet() )
					for( double[] b : c.get(i).get(a).keySet() ) {
						double cost = c.get(i).get(a).get(b);
						
						if( cost < bestCost ) {
							bestCost = cost;
							idx = i;
							na = a;
							nb = b;
						}
					}
	
			// add edge
			sub.get(idx).get(na).add(nb); // add new edge
			if (!sub.get(idx).containsKey(nb))
				sub.get(idx).put(nb, new HashSet<double[]>());
			sub.get(idx).get(nb).add(na);
			
			// update node states
			if( treeA.get(na).contains(nb) ) { // edge was from treeA
				if( ns.get(na) == NodeState.None )
					ns.put(na,NodeState.A);
				else if( ns.get(na) == NodeState.B )
					ns.put(na,NodeState.Both);
				
				if( ns.get(nb) == NodeState.None )
					ns.put(nb,NodeState.A);
				else if( ns.get(nb) == NodeState.B )
					ns.put(nb,NodeState.Both);	
			} 
			if( treeB.get(na).contains(nb)) { // edge was from treeB
				if( ns.get(na) == NodeState.None )
					ns.put(na,NodeState.B);
				else if( ns.get(na) == NodeState.A )
					ns.put(na,NodeState.Both);
				
				if( ns.get(nb) == NodeState.None )
					ns.put(nb,NodeState.B);
				else if( ns.get(nb) == NodeState.A )
					ns.put(nb,NodeState.Both);	
			}

			// check if we must merge idx with some sub
			for (int i = 0; i < sub.size(); i++)
				if (i != idx && added.get(i).contains(nb)) {
					for (Entry<double[], Set<double[]>> e : sub.get(i).entrySet())
						if (sub.get(idx).containsKey(e.getKey()))
							sub.get(idx).get(e.getKey()).addAll(e.getValue());
						else
							sub.get(idx).put(e.getKey(), e.getValue());
					sub.remove(i);

					added.get(idx).addAll(added.get(i));
					added.remove(i);
					break;
				}
		}
		nTree = new HashMap<double[], Set<double[]>>();
		for (Map<double[], Set<double[]>> m : sub)
			nTree.putAll(m);

		Map<double[], Set<double[]>> nCuts = new HashMap<double[], Set<double[]>>();
		Map<double[], Set<double[]>> cutsA = cuts;
		Map<double[], Set<double[]>> cutsB = mother.getCuts();

		// cut-candidates from cutsA/cutsB that can be used for nCuts
		Map<double[], Set<double[]>> c = new HashMap<double[], Set<double[]>>();
		for (double[] a : cutsA.keySet())
			for (double[] b : cutsA.get(a))
				if (nTree.containsKey(a) && nTree.get(a).contains(b)) {
					if (!c.containsKey(a))
						c.put(a, new HashSet<double[]>());
					c.get(a).add(b);
				}
		for (double[] a : cutsB.keySet())
			for (double[] b : cutsB.get(a))
				if (nTree.containsKey(a) && nTree.get(a).contains(b)) {
					if (!c.containsKey(a))
						c.put(a, new HashSet<double[]>());
					c.get(a).add(b);
				}

		// first, add cuts that they have in common (intersection)
		for (double[] a : c.keySet())
			for (double[] b : c.keySet())
				if (cutsA.containsKey(a) && cutsA.get(a).contains(b) && cutsB.containsKey(a) && cutsB.get(a).contains(b)) {
					if (!nCuts.containsKey(a))
						nCuts.put(a, new HashSet<double[]>());
					nCuts.get(a).add(b);
				}

		// remove added cuts from candidates
		for (double[] a : nCuts.keySet()) {
			if (c.containsKey(a)) {
				for (double[] b : nCuts.get(a))
					if (c.get(a).contains(b))
						c.get(a).remove(b);
				if (c.get(a).isEmpty())
					c.remove(a);
			}
		}

		int numCuts = countEdges(cuts);
		while (countEdges(nCuts) < numCuts) { // always to few cuts
			if (!c.isEmpty()) { // add random cut
				double[] ra = new ArrayList<double[]>(c.keySet()).get(r.nextInt(c.keySet().size()));
				double[] rb = new ArrayList<double[]>(c.get(ra)).get(r.nextInt(c.get(ra).size()));
				if (!nCuts.containsKey(ra))
					nCuts.put(ra, new HashSet<double[]>());
				nCuts.get(ra).add(rb);
				c.get(ra).remove(rb);
				if (c.get(ra).isEmpty())
					c.remove(ra);

				if (!nCuts.containsKey(rb))
					nCuts.put(rb, new HashSet<double[]>());
				nCuts.get(rb).add(ra);
				c.get(rb).remove(ra);
				if (c.get(rb).isEmpty())
					c.remove(rb);
			} else { // still to few cuts, happens really rarely
				double[] ra = new ArrayList<double[]>(nTree.keySet()).get(r.nextInt(nTree.keySet().size()));
				double[] rb = new ArrayList<double[]>(nTree.get(ra)).get(r.nextInt(nTree.get(ra).size()));

				if (!nCuts.containsKey(ra))
					nCuts.put(ra, new HashSet<double[]>());
				nCuts.get(ra).add(rb);

				if (!nCuts.containsKey(rb))
					nCuts.put(rb, new HashSet<double[]>());
				nCuts.get(rb).add(ra);
			}
		}
		return new TreeIndividual2(cm, nTree, nCuts);
	}
}
