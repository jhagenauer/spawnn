package regionalization.nga;

import java.util.AbstractMap.SimpleEntry;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import spawnn.dist.ConstantDist;
import spawnn.utils.GraphUtils;

/* mutation of tree only within distance k or tournament selection*/
public class TreeIndividual2 extends TreeIndividual {
	
	public TreeIndividual2(Map<double[], Set<double[]>> cm, Map<double[], Set<double[]>> tree, int numCluster) {
		super(cm,tree,numCluster);
	}

	public TreeIndividual2(Map<double[], Set<double[]>> cm, Map<double[], Set<double[]>> tree, Map<double[], Set<double[]>> cuts) {
		super(cm,tree,cuts);
	}
	
	public static int k = 1;
	public enum mode {min,max,sum};
	public static mode m = mode.min;

	@Override
	public void mutate() {
		int numCuts = countEdges(cuts);

		boolean f = new Random().nextBoolean();
		if ( (f && !onlyMutCuts) || onlyMutTrees ) {
			// remove random edge
			double[] ra = new ArrayList<double[]>(tree.keySet()).get(r.nextInt(tree.keySet().size()));
			double[] rb = new ArrayList<double[]>(tree.get(ra)).get(r.nextInt(tree.get(ra).size()));

			tree.get(ra).remove(rb);
			tree.get(rb).remove(ra);
			List<Map<double[], Set<double[]>>> sub = GraphUtils.getSubGraphs(tree);

			if (sub.size() != 2)
				throw new RuntimeException("Not a tree!");
			
			Map<double[], Double> m0, m1;
			if(  GraphUtils.getNodes(sub.get(0)).contains(ra) ) {
					m0 = GraphUtils.getShortestDists( GraphUtils.toWeightedGraph(sub.get(0), new ConstantDist<>(1.0)), ra);
					m1 = GraphUtils.getShortestDists( GraphUtils.toWeightedGraph(sub.get(1), new ConstantDist<>(1.0)), rb);
			} else {
				m0 = GraphUtils.getShortestDists( GraphUtils.toWeightedGraph(sub.get(0), new ConstantDist<>(1.0)), rb);
				m1 = GraphUtils.getShortestDists( GraphUtils.toWeightedGraph(sub.get(1), new ConstantDist<>(1.0)), ra);
			}
			
			// get candidates for new edge
			Map<double[], Map<double[],Double>> c = new HashMap<double[], Map<double[],Double>>();
			for (double[] a : sub.get(0).keySet()) {
				Map<double[],Double> s = new HashMap<double[],Double>();
				for (double[] b : sub.get(1).keySet())
					if (cm.containsKey(a) && cm.get(a).contains(b) ) {
						if( m == mode.min )
							s.put(b, Math.max(m0.get(a),m1.get(b) )+1 ); // min, max or sum?
						else if( m == mode.max )
							s.put(b, Math.min(m0.get(a),m1.get(b) )+1 ); 
						else if( m == mode.sum )
							s.put(b, m0.get(a) + m1.get(b) +1 ); 
					}
				if (!s.isEmpty())
					c.put(a, s);
			}
						
			Entry<double[],double[]> n = selectEdgesByCost(c, k);
			double[] na = n.getKey();
			double[] nb = n.getValue();

			// add new random edge
			//double[] na = new ArrayList<double[]>(c.keySet()).get(r.nextInt(c.keySet().size()));
			//double[] nb = new ArrayList<double[]>(c.get(na).keySet()).get(r.nextInt(c.get(na).size()));
			
			if (!tree.containsKey(na))
				tree.put(na, new HashSet<double[]>());
			tree.get(na).add(nb);
			if (!tree.containsKey(nb))
				tree.put(nb, new HashSet<double[]>());
			tree.get(nb).add(na);

			// removed edge was a cut-edge
			if (cuts.containsKey(ra) && cuts.get(ra).contains(rb)) {
				// remove old cut
				cuts.get(ra).remove(rb);
				if (cuts.get(ra).isEmpty())
					cuts.remove(ra);

				cuts.get(rb).remove(ra);
				if (cuts.get(rb).isEmpty())
					cuts.remove(rb);

				// add new cut
				if (!cuts.containsKey(na))
					cuts.put(na, new HashSet<double[]>());
				cuts.get(na).add(nb);

				if (!cuts.containsKey(nb))
					cuts.put(nb, new HashSet<double[]>());
				cuts.get(nb).add(na);
			}
		}
		if (numCuts > 0 && ((!f && !onlyMutTrees) || onlyMutCuts ) ) {

			// get potentital new cuts
			Map<double[], Set<double[]>> c = new HashMap<double[], Set<double[]>>();
			for (double[] a : tree.keySet()) {
				Set<double[]> s = new HashSet<double[]>();
				for (double[] b : tree.get(a))
					if (!cuts.containsKey(a) || !cuts.get(a).contains(b))
						s.add(b);
				if (!s.isEmpty())
					c.put(a, s);
			}

			if (!c.isEmpty()) {
				// randomly remove one cut
				double[] na = new ArrayList<double[]>(cuts.keySet()).get(r.nextInt(cuts.keySet().size()));
				double[] nb = new ArrayList<double[]>(cuts.get(na)).get(r.nextInt(cuts.get(na).size()));

				cuts.get(na).remove(nb);
				if (cuts.get(na).isEmpty())
					cuts.remove(na);
				cuts.get(nb).remove(na);
				if (cuts.get(nb).isEmpty())
					cuts.remove(nb);

				// randomly add one new cut
				double[] ra = new ArrayList<double[]>(c.keySet()).get(r.nextInt(c.keySet().size()));
				double[] rb = new ArrayList<double[]>(c.get(ra)).get(r.nextInt(c.get(ra).size()));
				if (!cuts.containsKey(ra))
					cuts.put(ra, new HashSet<double[]>());
				cuts.get(ra).add(rb);
				if (!cuts.containsKey(rb))
					cuts.put(rb, new HashSet<double[]>());
				cuts.get(rb).add(ra);
			}
		}

		if (numCuts != countEdges(cuts))
			throw new RuntimeException("Wrong number of cuts!");
	}

	@Override
	public TreeIndividual recombine(TreeIndividual mother) {

		Map<double[], Set<double[]>> nTree = new HashMap<double[], Set<double[]>>();
		Map<double[], Set<double[]>> treeA = tree;
		Map<double[], Set<double[]>> treeB = mother.getTree();

		for (double[] a : treeA.keySet()) // init with all nodes
			nTree.put(a, new HashSet<double[]>());

		for (double[] a : treeA.keySet()) // add intersections
			for (double[] b : treeA.get(a))
				if (treeB.containsKey(a) && treeB.get(a).contains(b)) {
					if (!nTree.containsKey(a))
						nTree.put(a, new HashSet<double[]>());
					nTree.get(a).add(b);
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
					
					belongs assigned = belongs.None; // node belongs to none exclusively
					for (double[] b : sub.get(i).get(a))
						if( treeA.get(a).contains(b) ) {
							assigned = belongs.A; 
							break;
						}
					for (double[] b : sub.get(i).get(a))
						if( treeB.get(a).contains(b) ) {
							if( assigned == belongs.None )
								assigned = belongs.B;
							else
								assigned = belongs.None; 
							break;
						}
					
					Map<double[],Double> s = new HashMap<double[],Double>();
					for (double[] b : treeA.get(a))
						if (!added.get(i).contains(b)) // avoid cycles
							s.put(b, assigned == belongs.B ? 10 : r.nextDouble() );
					
					for (double[] b : treeB.get(a))
						if (!added.get(i).contains(b)) // avoid cycles
							s.put(b, assigned == belongs.A ? 10 : r.nextDouble() );
					
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
	
	public Entry<double[],double[]> selectEdgesByCost(final Map<double[], Map<double[], Double>> c, int k) {
		List<Entry<double[], double[]>> l = new ArrayList<Entry<double[], double[]>>();
		for (double[] a : c.keySet())
			for (double[] b : c.get(a).keySet() ) {
				Entry<double[], double[]> e = new SimpleEntry<double[], double[]>(a, b);
				l.add(e);
			}
		
		if( k == -1 ) { // tournament selection
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for (Entry<double[], double[]> e : l) {
				double w = c.get(e.getKey()).get(e.getValue());
				if (w <= 0)
					throw new RuntimeException("value " + w + " not allowed in wGraph!");
				ds.addValue(w);
			}
			double sum = 0;
			for (double d : ds.getValues())
				sum += Math.abs(d - ds.getMax() + ds.getMin());

			double v = r.nextDouble() * sum;
			double lower = 0;
			for (Entry<double[], double[]> e : l) {
				double w = Math.abs(c.get(e.getKey()).get(e.getValue()) - ds.getMax() + ds.getMin());
				if (lower <= v && v <= lower + w) 
					return new SimpleEntry<double[],double[]>(e.getKey(),e.getValue());
				lower += w;
			}
		} else if( k > 0 ) { // select random from k-nearest
			Collections.sort(l, new Comparator<Entry<double[], double[]>>() {
				@Override
				public int compare(Entry<double[], double[]> o1, Entry<double[], double[]> o2) {
					double w1 = c.get(o1.getKey()).get(o1.getValue());
					double w2 = c.get(o2.getKey()).get(o2.getValue());
					return Double.compare(w1, w2);
				}
			});

			Map<double[], Set<double[]>> rc = new HashMap<double[], Set<double[]>>();
			for (Entry<double[], double[]> e : l.subList(0, Math.min(k, l.size()))) {
				if (!rc.containsKey(e.getKey()))
					rc.put(e.getKey(), new HashSet<double[]>());
				rc.get(e.getKey()).add(e.getValue());
			}
			double[] ra = new ArrayList<double[]>(rc.keySet()).get(r.nextInt(rc.keySet().size()));
			double[] rb = new ArrayList<double[]>(rc.get(ra)).get(r.nextInt(rc.get(ra).size()));
			return new SimpleEntry<double[],double[]>(ra,rb);
		}
		return null;
	}
}
