package regionalization.nga.tabu;

import java.util.AbstractMap.SimpleEntry;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import heuristics.tabu.TabuIndividual;
import heuristics.tabu.TabuMove;
import regionalization.nga.TreeIndividual;
import spawnn.dist.ConstantDist;
import spawnn.utils.GraphUtils;

public class CutsTabuIndividual extends TabuIndividual<CutsTabuIndividual> {

	public static int k = 0;
	public static int rndMoves = 10;

	Random r = new Random();
	Map<double[], Set<double[]>> tree, cuts;

	public CutsTabuIndividual(Map<double[], Set<double[]>> tree, Map<double[], Set<double[]>> cuts) {
		this.tree = tree;
		this.cuts = cuts;
	}
	
	public CutsTabuIndividual(Map<double[], Set<double[]>> tree, int numCluster ) {
		this.tree = tree;
		this.cuts = new HashMap<double[], Set<double[]>>();
		int numCuts = 0;
		while (numCuts < numCluster - 1) {
			double[] na = new ArrayList<double[]>(tree.keySet()).get(r.nextInt(tree.keySet().size()));
			double[] nb = new ArrayList<double[]>(tree.get(na)).get(r.nextInt(tree.get(na).size()));

			if (!cuts.containsKey(na) || !cuts.get(na).contains(nb)) {
				if (!cuts.containsKey(na))
					cuts.put(na, new HashSet<double[]>());
				cuts.get(na).add(nb);

				if (!cuts.containsKey(nb))
					cuts.put(nb, new HashSet<double[]>());
				cuts.get(nb).add(na);
				numCuts++;
			}
		}
	}

	@Override
	public CutsTabuIndividual applyMove(TabuMove<CutsTabuIndividual> tm) {

		Map<double[], Set<double[]>> nCuts = new HashMap<double[], Set<double[]>>();
		for (Entry<double[], Set<double[]>> e : cuts.entrySet())
			nCuts.put(e.getKey(), new HashSet<double[]>(e.getValue()));

		CutsTabuMove<CutsTabuIndividual> m = (CutsTabuMove<CutsTabuIndividual>)tm;
		Entry<double[], double[]> oldCut = m.getOldCut();
		Entry<double[], double[]> newCut = m.getNewCut();

		// remove old
		nCuts.get(oldCut.getKey()).remove(oldCut.getValue());
		if (nCuts.get(oldCut.getKey()).isEmpty())
			nCuts.remove(oldCut.getKey());
		nCuts.get(oldCut.getValue()).remove(oldCut.getKey());
		if (nCuts.get(oldCut.getValue()).isEmpty())
			nCuts.remove(oldCut.getValue());

		// add new cut
		if (!nCuts.containsKey(newCut.getKey()))
			nCuts.put(newCut.getKey(), new HashSet<double[]>());
		nCuts.get(newCut.getKey()).add(newCut.getValue());
		if (!nCuts.containsKey(newCut.getValue()))
			nCuts.put(newCut.getValue(), new HashSet<double[]>());
		nCuts.get(newCut.getValue()).add(newCut.getKey());

		return new CutsTabuIndividual(tree, nCuts);
	}

	@Override
	public TabuMove<CutsTabuIndividual> getRandomMove() {
		// randomly remove one cut
		double[] na = new ArrayList<double[]>(cuts.keySet()).get(r.nextInt(cuts.keySet().size()));
		double[] nb = new ArrayList<double[]>(cuts.get(na)).get(r.nextInt(cuts.get(na).size()));

		cuts.get(na).remove(nb);
		if (cuts.get(na).isEmpty())
			cuts.remove(na);
		cuts.get(nb).remove(na);
		if (cuts.get(nb).isEmpty())
			cuts.remove(nb);

		Map<double[], Double> m0 = GraphUtils.getShortestDists(GraphUtils.toWeightedGraph(tree, new ConstantDist<>(1.0)), na);
		Map<double[], Double> m1 = GraphUtils.getShortestDists(GraphUtils.toWeightedGraph(tree, new ConstantDist<>(1.0)), nb);
		// get candidates for new edge
		Map<double[], Map<double[], Double>> c = new HashMap<double[], Map<double[], Double>>();
		for (double[] a : tree.keySet()) {
			Map<double[], Double> s = new HashMap<double[], Double>();
			for (double[] b : tree.get(a))
				if (!cuts.containsKey(a) || !cuts.get(a).contains(b)) { // no double-cuts
					s.put(b, Math.min(m0.get(a) + m1.get(b), m0.get(b) + m1.get(a)) + 1);
				}
			if (!s.isEmpty())
				c.put(a, s);
		}

		if (!cuts.containsKey(na))
			cuts.put(na, new HashSet<double[]>());
		cuts.get(na).add(nb);
		if (!cuts.containsKey(nb))
			cuts.put(nb, new HashSet<double[]>());
		cuts.get(nb).add(na);

		return new CutsTabuMove<>(new SimpleEntry<double[], double[]>(na, nb), TreeIndividual.selectEdgesByCost(c, k));
	}

	@Override
	public List<TabuMove<CutsTabuIndividual>> getNeighboringMoves() {
		if( k != 0 ) {
			Set<TabuMove<CutsTabuIndividual>> s = new HashSet<>();
			while( s.size() != rndMoves ) 
				s.add( getRandomMove() );
			return new ArrayList<>(s);
		}
		Set<Entry<double[],double[]>> oldCuts = new HashSet<Entry<double[],double[]>>(); 
		for (double[] a : cuts.keySet())
			for (double[] b : cuts.get(a)) {
				Entry<double[],double[]> e = new SimpleEntry<>(a,b);
				Entry<double[],double[]> er = new SimpleEntry<>(b,a);
				if( !oldCuts.contains(er) )
					oldCuts.add(e);
			}
		
		Map<Entry<double[],double[]>,Set<Entry<double[],double[]>>> c = new HashMap<>();
		for( Entry<double[],double[]> e : oldCuts ) {
			
			for( double[] nb : tree.get(e.getKey() ) ) {
				Entry<double[],double[]> n = new SimpleEntry<double[],double[]>(e.getKey(),nb);
				Entry<double[],double[]> nr = new SimpleEntry<double[],double[]>(nb,e.getKey());
				if( !c.containsKey(e) )
					c.put(e, new HashSet<>() );
				if( !c.get(e).contains(n) && !c.get(e).contains(nr) && !oldCuts.contains(n) && !oldCuts.contains(nr)) 
					c.get(e).add(n);
					
			}
			for( double[] nb : tree.get(e.getValue() ) ) {
				Entry<double[],double[]> n = new SimpleEntry<double[],double[]>(e.getValue(),nb);
				Entry<double[],double[]> nr = new SimpleEntry<double[],double[]>(nb,e.getValue());
				if( !c.containsKey(e) )
					c.put(e, new HashSet<>() );
				if( !c.get(e).contains(n) && !c.get(e).contains(nr) && !oldCuts.contains(n) && !oldCuts.contains(nr) ) 
					c.get(e).add(n);		
			}
		}

		List<TabuMove<CutsTabuIndividual>> l = new ArrayList<TabuMove<CutsTabuIndividual>>();
		for( Entry<Entry<double[],double[]>,Set<Entry<double[],double[]>>> e : c.entrySet() )
			for( Entry<double[], double[]> s : e.getValue() )
				l.add( new CutsTabuMove<>(e.getKey(), s));
		
		return l;
	}
	
	public List<Set<double[]>> toCluster() {
		Map<double[], Set<double[]>> cutTree = new HashMap<double[], Set<double[]>>();

		for (Entry<double[], Set<double[]>> e : tree.entrySet()) {
			Set<double[]> s = new HashSet<double[]>(e.getValue());
			if (cuts.containsKey(e.getKey()))
				s.removeAll(cuts.get(e.getKey()));
			cutTree.put(e.getKey(), s);
		}

		List<Set<double[]>> clusters = new ArrayList<Set<double[]>>();
		for (Map<double[], Set<double[]>> sg : GraphUtils.getSubGraphs(cutTree)) {
			Set<double[]> nodes = new HashSet<double[]>(sg.keySet());
			for (double[] a : sg.keySet())
				nodes.addAll(sg.get(a));
			clusters.add(nodes);
		}
		return clusters;
	}

	public Map<double[], Set<double[]>> getTree() {
		return tree;
	}

	public Map<double[], Set<double[]>> getCuts() {
		return cuts;
	}

}
