package regionalization.nga;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import myga.GAIndividual;
import spawnn.utils.Clustering;
import spawnn.utils.Drawer;

public class TreeIndividual extends GAIndividual<TreeIndividual> {
	Random r = new Random();
	Map<double[], Set<double[]>> cm, tree, cuts;

	public TreeIndividual(Map<double[], Set<double[]>> cm, Map<double[], Set<double[]>> tree) {
		this.cm = cm;
		this.tree = tree;
		this.cuts = new HashMap<double[], Set<double[]>>();
	}

	public TreeIndividual(Map<double[], Set<double[]>> cm, Map<double[], Set<double[]>> tree, Map<double[], Set<double[]>> cuts) {
		this.cm = cm;
		this.tree = tree;
		this.cuts = cuts;
	}

	@Override
	public void mutate() {
		int numCuts = countEdges(cuts);
		{
			// remove random edge
			double[] ra = new ArrayList<double[]>(tree.keySet()).get(r.nextInt(tree.keySet().size()));
			double[] rb = new ArrayList<double[]>(tree.get(ra)).get(r.nextInt(tree.get(ra).size()));
			tree.get(ra).remove(rb);
			tree.get(rb).remove(ra);
			
			// remove cut edge was a cut-edge
			if( cuts.containsKey(ra) && cuts.get(ra).contains(rb) ) {
				cuts.get(ra).remove(rb);
				if (cuts.get(ra).isEmpty())
					cuts.remove(ra);
			
				cuts.get(rb).remove(ra);
				if (cuts.get(rb).isEmpty())
					cuts.remove(rb);
			}
			
			Map<double[],Set<double[]>> conn = new HashMap<double[],Set<double[]>>();
			List<Map<double[], Set<double[]>>> sub = Clustering.getSubGraphs(tree);
			
			if( sub.size() != 2 ) {
				System.out.println(sub.size());
				
				Map<double[],double[]> hl = new HashMap<double[],double[]>();
				hl.put(ra, rb);
				Drawer.geoDrawConnections(tree, hl, new int[]{0,1}, null, "output/tree.png");
				System.exit(1);
			}
			
			for( double[] a : sub.get(0).keySet() ) {
				Set<double[]> s = new HashSet<double[]>();
				for( double[] b : sub.get(1).keySet() ) 
					if( cm.containsKey(a) && cm.get(a).contains(b) )
						s.add(b);
				if( !s.isEmpty() )
					conn.put(a, s);
			}
					
			double[] na = new ArrayList<double[]>(conn.keySet()).get(r.nextInt(conn.keySet().size()));
			double[] nb = new ArrayList<double[]>(conn.get(na)).get(r.nextInt(conn.get(na).size()));

			if (!tree.containsKey(na))
				tree.put(na, new HashSet<double[]>());
			tree.get(na).add(nb);
			if (!tree.containsKey(nb))
				tree.put(nb, new HashSet<double[]>());
			tree.get(nb).add(na);
		}
		
		if( numCuts > 0 && numCuts == countEdges(cuts) ){ // randomly remove one cut
			double[] na = new ArrayList<double[]>(cuts.keySet()).get(r.nextInt(cuts.keySet().size()));
			double[] nb = new ArrayList<double[]>(cuts.get(na)).get(r.nextInt(cuts.get(na).size()));

			cuts.get(na).remove(nb);
			if (cuts.get(na).isEmpty())
				cuts.remove(na);
			cuts.get(nb).remove(na);
			if (cuts.get(nb).isEmpty())
				cuts.remove(nb);
		}

		// add one new cut
		while (countEdges(cuts) < numCuts) {
			double[] na = new ArrayList<double[]>(tree.keySet()).get(r.nextInt(tree.keySet().size()));
			double[] nb = new ArrayList<double[]>(tree.get(na)).get(r.nextInt(tree.get(na).size()));
			if (!cuts.containsKey(na))
				cuts.put(na, new HashSet<double[]>());
			cuts.get(na).add(nb);

			if (!cuts.containsKey(nb))
				cuts.put(nb, new HashSet<double[]>());
			cuts.get(nb).add(na);
		}
	}

	@Override
	public TreeIndividual recombine(TreeIndividual mother) {
		Map<double[], Set<double[]>> treeA = tree;
		Map<double[], Set<double[]>> treeB = mother.getTree();


		Map<double[], Set<double[]>> nTree = new HashMap<double[], Set<double[]>>();
		
		// get random points (Maybe it would be better to make sure that the random points have more than one neighbor)
		boolean even = true;
		boolean aFull = false, bFull = false;
		double[] startA = new ArrayList<double[]>(treeA.keySet()).get(r.nextInt(treeA.keySet().size()));
		double[] startB = null;
		do {
			startB = new ArrayList<double[]>(treeB.keySet()).get(r.nextInt(treeB.keySet().size()));
		} while( startA == startB || treeA.get(startA).contains(startB) || treeB.get(startB).contains(startA) ); // neccesary?
		
		Set<double[]> addedA = new HashSet<double[]>();
		nTree.put(startA,new HashSet<double[]>());
		addedA.add(startA);
		Set<double[]> addedB = new HashSet<double[]>();
		nTree.put(startB, new HashSet<double[]>());
		addedB.add(startB);

		// add nodes that belong exclusively to the subtrees
		while (!aFull || !bFull) {
			if (even && !aFull) { // from treeA
				Map<double[], Set<double[]>> c = new HashMap<double[], Set<double[]>>();
				for (double[] a : addedA) {
					Set<double[]> s = new HashSet<double[]>();
					for (double[] b : treeA.get(a))
						if (!addedA.contains(b) && !addedB.contains(b))
							s.add(b);
					if (!s.isEmpty())
						c.put(a, s);
				}
				if (!c.isEmpty()) {
					double[] na = new ArrayList<double[]>(c.keySet()).get(r.nextInt(c.keySet().size()));
					double[] nb = new ArrayList<double[]>(c.get(na)).get(r.nextInt(c.get(na).size()));

					// add connections to both directions
					if (!nTree.containsKey(na))
						nTree.put(na, new HashSet<double[]>());
					nTree.get(na).add(nb);
					if (!nTree.containsKey(nb))
						nTree.put(nb, new HashSet<double[]>());
					nTree.get(nb).add(na);
					addedA.add(nb);
				} else
					aFull = true;
			} else if (!bFull) { // from treeB
				Map<double[], Set<double[]>> c = new HashMap<double[], Set<double[]>>();
				for (double[] a : addedB) {
					Set<double[]> s = new HashSet<double[]>();
					for (double[] b : treeB.get(a))
						if (!addedA.contains(b) && !addedB.contains(b))
							s.add(b);
					if (!s.isEmpty())
						c.put(a, s);
				}
				if (!c.isEmpty()) {
					double[] na = new ArrayList<double[]>(c.keySet()).get(r.nextInt(c.keySet().size()));
					double[] nb = new ArrayList<double[]>(c.get(na)).get(r.nextInt(c.get(na).size()));

					// add connections to both directions
					if (!nTree.containsKey(na))
						nTree.put(na, new HashSet<double[]>());
					nTree.get(na).add(nb);
					if (!nTree.containsKey(nb))
						nTree.put(nb, new HashSet<double[]>());
					nTree.get(nb).add(na);
					addedB.add(nb);
				} else
					bFull = true;
			}
			even = !even;
		}
				
		// connect subtrees
		Map<double[], Set<double[]>> conn = new HashMap<double[], Set<double[]>>();
		for (double[] a : addedA) {
			Set<double[]> s = new HashSet<double[]>();
			for (double[] b : addedB)
				if (treeA.get(a).contains(b))
					s.add(b);
			if (!s.isEmpty())
				conn.put(a, s);
		}

		double[] na = new ArrayList<double[]>(conn.keySet()).get(r.nextInt(conn.keySet().size()));
		double[] nb = new ArrayList<double[]>(conn.get(na)).get(r.nextInt(conn.get(na).size()));

		if (!nTree.containsKey(na))
			nTree.put(na, new HashSet<double[]>());
		nTree.get(na).add(nb);
		if (!nTree.containsKey(nb))
			nTree.put(nb, new HashSet<double[]>());
		nTree.get(nb).add(na);
				
		// build complete spanning tree, prefer edges from treeA and treeB
		Set<double[]> added = new HashSet<double[]>(addedA);
		added.addAll(addedB);
		while (added.size() != cm.size()) {

			Map<double[], Set<double[]>> fromTrees = new HashMap<double[], Set<double[]>>();
			Map<double[], Set<double[]>> nEdges = new HashMap<double[], Set<double[]>>();
			for (double[] a : added) {
				Set<double[]> s1 = new HashSet<double[]>();
				for (double[] b : treeA.get(a))
					if (!added.contains(b))
						s1.add(b);
				for (double[] b : treeB.get(a))
					if (!added.contains(b))
						s1.add(b);
				if (!s1.isEmpty())
					fromTrees.put(a, s1);

				Set<double[]> s2 = new HashSet<double[]>();
				for (double[] b : cm.get(a))
					if (!added.contains(b) && s1.contains(b))
						s2.add(b);
				if (!s2.isEmpty())
					nEdges.put(a, s2);
			}

			double[] ra;
			double[] rb;
			if (!fromTrees.isEmpty()) {
				ra = new ArrayList<double[]>(fromTrees.keySet()).get(r.nextInt(fromTrees.keySet().size()));
				rb = new ArrayList<double[]>(fromTrees.get(ra)).get(r.nextInt(fromTrees.get(ra).size()));
			} else { // nEdges should never be empty
				ra = new ArrayList<double[]>(nEdges.keySet()).get(r.nextInt(nEdges.keySet().size()));
				rb = new ArrayList<double[]>(nEdges.get(ra)).get(r.nextInt(nEdges.get(ra).size()));
			}

			// add connections to both directions
			if (!nTree.containsKey(ra))
				nTree.put(ra, new HashSet<double[]>());
			nTree.get(ra).add(rb);

			if (!nTree.containsKey(rb))
				nTree.put(rb, new HashSet<double[]>());
			nTree.get(rb).add(ra);

			added.add(rb);
		}

		Map<double[], Set<double[]>> nCuts = new HashMap<double[], Set<double[]>>();
		// new cuts
		Map<double[], Set<double[]>> cutsA = cuts;
		Map<double[], Set<double[]>> cutsB = mother.getCuts();
		for (double[] a : nTree.keySet())
			for (double[] b : nTree.get(a)) {
				if (cutsA.containsKey(a) && cutsA.get(a).contains(b)) {
					if (!nCuts.containsKey(a))
						nCuts.put(a, new HashSet<double[]>());
					nCuts.get(a).add(b);
				}
				if (cutsB.containsKey(b) && cutsB.get(b).contains(a)) {
					if (!nCuts.containsKey(b))
						nCuts.put(b, new HashSet<double[]>());
					nCuts.get(b).add(a);
				}
			}

		// repair
		int numCuts = countEdges(cuts);
		// to many cuts
		while (countEdges(nCuts) > numCuts) {
			double[] ra = new ArrayList<double[]>(nCuts.keySet()).get(r.nextInt(nCuts.keySet().size()));
			double[] rb = new ArrayList<double[]>(nCuts.get(ra)).get(r.nextInt(nCuts.get(ra).size()));

			nCuts.get(ra).remove(rb);
			nCuts.get(rb).remove(ra);
			if (nCuts.get(ra).isEmpty())
				nCuts.remove(ra);
			if (nCuts.get(rb).isEmpty())
				nCuts.remove(rb);
		}

		// to few cuts
		while (countEdges(nCuts) < numCuts) {
			double[] ra = new ArrayList<double[]>(nTree.keySet()).get(r.nextInt(nTree.keySet().size()));
			double[] rb = new ArrayList<double[]>(nTree.get(ra)).get(r.nextInt(nTree.get(ra).size()));
			if (!nCuts.containsKey(ra))
				nCuts.put(ra, new HashSet<double[]>());
			nCuts.get(ra).add(rb);

			if (!nCuts.containsKey(rb))
				nCuts.put(rb, new HashSet<double[]>());
			nCuts.get(rb).add(ra);
		}
				
		return new TreeIndividual(cm, nTree, nCuts);
	}

	public Map<double[], Set<double[]>> getTree() {
		return tree;
	}

	public Map<double[], Set<double[]>> getCuts() {
		return cuts;
	}

	private int countEdges(Map<double[], Set<double[]>> m) {
		int num = 0;
		for (Set<double[]> s : m.values())
			num += s.size();
		return num;
	}
	
	private static boolean hasCircles(Map<double[],Set<double[]>> tree ) {
		Map<double[],Set<double[]>> cm = getCopy(tree);
		
		for( double[] ra : tree.keySet() )
			for( double[] rb: tree.get(ra) ) {
				cm.get(ra).remove(rb);
				cm.get(rb).remove(ra);
				
				List<Map<double[], Set<double[]>>> sub = Clustering.getSubGraphs(cm);
				if( sub.size() != 2 )
					return true;
				cm.get(ra).add(rb);
				cm.get(rb).add(ra);
			}
		return false;
	}
	
	private static Map<double[],Set<double[]>> getCopy(Map<double[],Set<double[]>> tree) {
		Map<double[],Set<double[]>> cm = new HashMap<double[],Set<double[]>>();
		for( Entry<double[],Set<double[]>> e : tree.entrySet() )
			cm.put(e.getKey(), new HashSet<double[]>(e.getValue() ) );
		return cm;
	}
	

	public List<Set<double[]>> toCluster() {
		Map<double[],Set<double[]>> cutTree = new HashMap<double[],Set<double[]>>();
		
		for( Entry<double[],Set<double[]>> e : tree.entrySet() ) {
			Set<double[]> s = new HashSet<double[]>(e.getValue());
			if( cuts.containsKey(e.getKey() ))
				s.removeAll(cuts.get(e.getKey()));
			cutTree.put(e.getKey(), s);
		}
		
		List<Set<double[]>> clusters = new ArrayList<Set<double[]>>();
		for (Map<double[], Set<double[]>> sg : Clustering.getSubGraphs(cutTree)) {
			Set<double[]> nodes = new HashSet<double[]>(sg.keySet());
			for (double[] a : sg.keySet())
				nodes.addAll(sg.get(a));
			clusters.add(nodes);
		}
		return clusters;
	}
}
