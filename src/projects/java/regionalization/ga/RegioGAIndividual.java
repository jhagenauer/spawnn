package regionalization.ga;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.jfree.util.Log;

import spawnn.utils.ClusterValidation;

public class RegioGAIndividual implements GAIndividual {

	protected List<double[]> genome;
	protected int seedSize;
	final protected Map<double[], Set<double[]>> cm;
	protected final ClusterCostCalculator cc;

	protected List<Set<double[]>> cluster;
	protected double cost;
	public double probSeedGenMod;

	public List<Integer> bbs = new ArrayList<Integer>();

	private Random r = new Random();

	// Mai, Maltoni and Rizzi (1995): inspired by Topological clustering of maps using genetic algorithm
	public RegioGAIndividual( final List<double[]> genome, final int seedSize, ClusterCostCalculator cc, final Map<double[], Set<double[]>> cm) {
		this.seedSize = seedSize;
		this.cm = cm;
		this.cc = cc;
		this.genome = genome;

		this.cluster = decode();
		this.cost = getCost(this.cluster);
		
		probSeedGenMod = -1;//(double)seedSize/genome.size() ;//-1;
	}
	
	private List<Set<double[]>> decode() {
		// stores cost calculations for speed up
		Map<Integer, Double> costMap = new HashMap<Integer, Double>();

		// decode seed
		List<Set<double[]>>cluster = new ArrayList<Set<double[]>>();
		for (int i = 0; i < seedSize; i++) {
			Set<double[]> s = new HashSet<double[]>();
			s.add(genome.get(i));
			cluster.add(s);

			costMap.put(i, cc.getCost(s));
		}

		List<double[]> nGenome = new ArrayList<double[]>(genome.subList(0, seedSize ));
		List<double[]> growth = new LinkedList<double[]>(genome.subList(seedSize, genome.size()));
		while (!growth.isEmpty()) {
			double[] cur = growth.remove(0);
			int bestIdx = -1;
			double lowestInc = Double.NaN;

			// check if cluster is adjacent to cur
			for (int idxS = 0; idxS < cluster.size(); idxS++) {
				Set<double[]> s = cluster.get(idxS);

				boolean connected = false;
				for (double[] nb : cm.get(cur)) {
					if (s.contains(nb)) {
						connected = true;
						break;
					}
				}
				
				if ( !connected )
					continue;

				s.add(cur);
				double inc = cc.getCost(s) - costMap.get(idxS);
				s.remove(cur);

				if (bestIdx < 0 || inc < lowestInc) {
					bestIdx = idxS;
					lowestInc = inc;
				}

			}

			if (bestIdx >= 0) {
				nGenome.add(cur);
				cluster.get(bestIdx).add(cur);
				costMap.put(bestIdx, costMap.get(bestIdx) + lowestInc);
			} else { // try later
				growth.add(cur);
			}
		}		
		
		this.costFromDecode = 0;
		for( int i = 0; i < seedSize; i++ )
			this.costFromDecode += costMap.get(i);
				
		this.genome = nGenome; // slightly improves the results ( 171.560 vs. 171.217 )
		
		// recode test, not useful probably
		/*nGenome = new ArrayList<double[]>(genome.subList(0, seedSize ));
		for (int i = 0; i < seedSize; i++) {
			double[] seed = genome.get(i);
			
			List<double[]> s = new ArrayList<double[]>();
			s.add(seed);
			
			Set<double[]> nGrowth = new HashSet<>(cluster.get(i));
			nGrowth.remove(seed);
			
			while( !nGrowth.isEmpty() ) {
								
				Set<double[]> nbs = new HashSet<>();
				for( double[] a : s )
					nbs.addAll( cm.get(a) );
				for( double[] a : nbs )
					if( nGrowth.contains(a) ) {
						nGrowth.remove(a);
						s.add(a);
					}			
			}
			s.remove(seed);
			nGenome.addAll(s);
		}
		
		for( double[] d : genome )
			for( int i = 0; i < cluster.size(); i++ )
				if( cluster.get(i).contains(d) )
					System.out.print(i);
		System.out.println();
		
		List<Set<double[]>> nCluster = toCluster(nGenome);
		
		for( double[] d : nGenome )
			for( int i = 0; i < nCluster.size(); i++ )
				if( nCluster.get(i).contains(d) )
					System.out.print(i);
		System.out.println();		
		
		System.out.println("nmi: "+ClusterValidation.getNormalizedMutualInformation(cluster, nCluster ));
				
		System.exit(1);
		genome = nGenome;*/
		
		return cluster;
	}
	
	// jsut for debug, derivate of decode
	private List<Set<double[]>> toCluster( List<double[]> g_nome ) {
		// stores cost calculations for speed up
		Map<Integer, Double> costMap = new HashMap<Integer, Double>();

		// decode seed
		List<Set<double[]>>cluster = new ArrayList<Set<double[]>>();
		for (int i = 0; i < seedSize; i++) {
			Set<double[]> s = new HashSet<double[]>();
			s.add(g_nome.get(i));
			cluster.add(s);

			costMap.put(i, cc.getCost(s));
		}

		List<double[]> growth = new LinkedList<double[]>(g_nome.subList(seedSize, g_nome.size()));
		while (!growth.isEmpty()) {
			double[] cur = growth.remove(0);
			int bestIdx = -1;
			double lowestInc = Double.NaN;

			// check if cluster is adjacent to cur
			for (int idxS = 0; idxS < cluster.size(); idxS++) {
				Set<double[]> s = cluster.get(idxS);

				boolean connected = false;
				for (double[] nb : cm.get(cur)) {
					if (s.contains(nb)) {
						connected = true;
						break;
					}
				}
				
				if ( !connected )
					continue;

				s.add(cur);
				double inc = cc.getCost(s) - costMap.get(idxS);
				s.remove(cur);

				if (bestIdx < 0 || inc < lowestInc) {
					bestIdx = idxS;
					lowestInc = inc;
				}

			}

			if (bestIdx >= 0) {
				cluster.get(bestIdx).add(cur);
				costMap.put(bestIdx, costMap.get(bestIdx) + lowestInc);
			} else { // try later
				growth.add(cur);
			}
		}			
		return cluster;
	}
	
	private double costFromDecode = 0;
	
	public double getCost( List<Set<double[]>> cluster ) {
		return costFromDecode;
		/*double cost = 0;
		for( Set<double[]> s : cluster )
			cost += cc.getCost(s);
		return cost;*/
	}

	public List<double[]> getGenome() {
		return genome;
	}

	public List<Set<double[]>> getCluster() {
		return cluster;
	}

	// NOTE: it is essential, that seed and growth exchange genes! Maios don't do that, if restricted
	// TODO: which one is better for complete genome?
	@Override
	public GAIndividual mutate() {
		if (probSeedGenMod < 0) {
			return maioMutate(0, genome.size());
			// return mutate(0, genome.size() );
		} else if (r.nextDouble() < probSeedGenMod) {
			return maioMutate(0, seedSize);
			// return mutate(0, seedSize );
		} else {
			return maioMutate(seedSize, genome.size() - seedSize);
			// return mutate(seedSize, genome.size()-seedSize );
		}
	}

	@Override
	public GAIndividual recombine(GAIndividual mother) {
		if (probSeedGenMod < 0) {
			return maioPartiallyMatched(mother, 0, genome.size());
			// return partiallyMatched(mother, 0, genome.size() );
		} else if (r.nextDouble() < probSeedGenMod) {
			return maioPartiallyMatched(mother, 0, seedSize);
			// return partiallyMatched(mother, 0, seedSize );
		} else {
			//return maioPartiallyMatched(mother, seedSize, genome.size() - seedSize);
			return partiallyMatched(mother, seedSize, genome.size() - seedSize);
		}
	}

	private GAIndividual mutate(int start, int size) {
		Random r = new Random();
		List<double[]> nGenome = new ArrayList<double[]>(genome);

		// exchange single element within length with one from complete genome
		for (int i = 0; i < size; i++) {
			if (r.nextDouble() < 1.0 / size) {
				int idxA = start + r.nextInt(size);
				int idxB = r.nextInt(nGenome.size());

				double[] valB = nGenome.set(idxB, nGenome.get(idxA));
				nGenome.set(idxA, valB);
			}
		}
		return new RegioGAIndividual(nGenome, seedSize, cc, cm);
	}

	private GAIndividual maioMutate(int start, int size) {
		Random r = new Random();
		List<double[]> nGenome = new ArrayList<double[]>(genome);

		for (int i = 0; i < size; i++) {
			if (r.nextDouble() < 1.0 / size) {

				int idxA = start + r.nextInt(size);
				int idxB = start + r.nextInt(size);

				double[] valB = nGenome.set(idxB, nGenome.get(idxA));
				nGenome.set(idxA, valB);
			}
		}
		return new RegioGAIndividual(nGenome, seedSize, cc, cm);
	}

	public GAIndividual partiallyMatched(GAIndividual mother, int start, int length) {
		Random r = new Random();
		List<double[]> mGenome = ((RegioGAIndividual) mother).getGenome();

		int idxA = start + r.nextInt(length - 1);
		int idxB = idxA + r.nextInt(start + length - idxA) + 1;

		List<double[]> nGenome = new ArrayList<double[]>(genome.size());
		for (int i = 0; i < genome.size(); i++)
			nGenome.add(null);

		// keep between idxA and idxB
		Set<double[]> keep = new HashSet<double[]>();
		for (int i = idxA; i < idxB; i++) {
			nGenome.set(i, genome.get(i));
			keep.add(genome.get(i));
		}

		// fill with mother
		for (int i = 0; i < genome.size(); i++)
			if (nGenome.get(i) == null && !keep.contains(mGenome.get(i)))
				nGenome.set(i, mGenome.get(i));

		// fill rest with genome
		List<double[]> rest = new LinkedList<double[]>(genome);
		rest.removeAll(nGenome);
		for (int i = genome.size() - 1; i >= 0; i--)
			if (nGenome.get(i) == null)
				nGenome.set(i, rest.remove(0));

		return new RegioGAIndividual(nGenome, seedSize, cc, cm);
	}

	public GAIndividual maioPartiallyMatched(GAIndividual mother, int start, int length) {
		Random r = new Random();
		List<double[]> mGenome = ((RegioGAIndividual) mother).getGenome();

		// partially matched cross over
		int idxA = start + r.nextInt(length - 1);
		int idxB = idxA + r.nextInt(start + length - idxA) + 1;

		List<double[]> nGenome = new ArrayList<double[]>(genome);

		for (int i = idxA; i < idxB; i++) {

			for (int j = 0; j < nGenome.size(); j++)
				if (nGenome.get(j) == mGenome.get(i)) {
					// swap i,j within nGenome
					double[] id = nGenome.set(i, nGenome.get(j));
					nGenome.set(j, id);
				}
		}
		return new RegioGAIndividual(nGenome, seedSize, cc, cm);
	}

	@Override
	public int compareTo(GAIndividual o) {
		if (getValue() < o.getValue())
			return -1;
		else if (getValue() > o.getValue())
			return 1;
		else
			return 0;
	}

	@Override
	public double getValue() {
		return cost;
	}
}
