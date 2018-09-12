package gwr.ga;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;

import gwr.ga.GWRIndividual;
import spawnn.dist.Dist;

public class GWRIndividualAdaptive implements GWRIndividual<GWRIndividualAdaptive> {

	protected Random r;

	protected List<Integer> chromosome;
	protected int minGene, maxGene;
	
	public static Map<Integer, Set<Integer>> cmI;

	public static double sd;

	public GWRIndividualAdaptive(List<Integer> chromosome, int minGene, int maxGene) {
		this.chromosome = chromosome;
		for (int i = 0; i < this.chromosome.size(); i++) {
			int h = this.chromosome.get(i);
			h = Math.max(minGene, Math.min(maxGene, h));
			this.chromosome.set(i, h);
		}
		this.minGene = minGene;
		this.maxGene = maxGene;
		this.r = new Random(chromosome.hashCode());
	}
			
	@Override
	public GWRIndividualAdaptive mutate() {
		List<Integer> nChromosome = new ArrayList<>();
		for (int j = 0; j < chromosome.size(); j++) {
			int h = chromosome.get(j);
			if (r.nextDouble() < 1.0 / chromosome.size()) {
				double d = r.nextGaussian()*sd;
				if( d < 0 )
					h += (int)Math.floor(d);
				else
					h += (int)Math.ceil(d);
				h = Math.max(minGene, Math.min(maxGene, h));
			}
			nChromosome.add(h);
		}
		return new GWRIndividualAdaptive(nChromosome, minGene, maxGene);
	}

	@Override
	public GWRIndividualAdaptive recombine(GWRIndividualAdaptive mother) {
		List<Integer> mChromosome = ((GWRIndividualAdaptive) mother).getChromosome();
		List<Integer> nChromosome = new ArrayList<>();

		for (int i = 0; i < chromosome.size(); i++) {
			if (r.nextBoolean())
				nChromosome.add(mChromosome.get(i));
			else
				nChromosome.add(chromosome.get(i));
		}
		
		GWRIndividualAdaptive i = new GWRIndividualAdaptive(nChromosome, minGene, maxGene);		
		return i;
	}

	public List<Integer> getChromosome() {
		return this.chromosome;
	}

	public int getGeneAt(int i) {
		return chromosome.get(i);
	}
	
	@Override
	public String toString() {
		return "min: " + Collections.min(chromosome) + " " +chromosome.subList(0,Math.min(chromosome.size(),30));
	}

	@Override
	public Map<double[], Double> getSpatialBandwidth(List<double[]> samples, Dist<double[]> gDist ) {
		Map<double[],Double> bandwidth = new HashMap<>();
		for (int i = 0; i < samples.size(); i++) {
			double[] a = samples.get(i);	
			int k = Math.min( samples.size(), chromosome.get(i) );			
				
			double[] b = getKthLargest( k, samples, new Comparator<double[]>() {
				@Override
				public int compare(double[] o1, double[] o2) {
					return -Double.compare(gDist.dist(o1, a), gDist.dist(o2, a));
				}
			});	
			bandwidth.put( a, gDist.dist(a, b) );			
		}	
		return bandwidth;
	}
	
	private double[] getKthLargest( int k, List<double[]> samples, Comparator<double[]> c) {
		PriorityQueue<double[]> q = new PriorityQueue<double[]>(k, c);
		for (double[] d : samples) {
			q.offer(d);
			if (q.size() > k)
				q.poll();
		}
		return q.peek();
	}

	@Override
	public String geneToString(int i) {
		return chromosome.get(i)+"";
	}
}
