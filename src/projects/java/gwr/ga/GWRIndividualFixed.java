package gwr.ga;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import spawnn.dist.Dist;

public class GWRIndividualFixed implements GWRIndividual<GWRIndividualFixed> {

	protected Random r;

	protected List<Double> chromosome;
	protected double minGene, maxGene;
	
	public static Map<Integer, Set<Integer>> cmI;

	public static double sd;

	public GWRIndividualFixed(List<Double> chromosome, double minGene, double maxGene) {
		this.chromosome = chromosome;
		for (int i = 0; i < this.chromosome.size(); i++) {
			double h = this.chromosome.get(i);
			h = Math.max(minGene, Math.min(maxGene, h));
			this.chromosome.set(i, h);
		}
		this.minGene = minGene;
		this.maxGene = maxGene;
		this.r = new Random(chromosome.hashCode());
	}
			
	@Override
	public GWRIndividualFixed mutate() {
		List<Double> nChromosome = new ArrayList<>();
		for (int j = 0; j < chromosome.size(); j++) {
			double h = chromosome.get(j);
			h += r.nextGaussian()*sd;
			nChromosome.add(h);
		}
		return new GWRIndividualFixed(nChromosome, minGene, maxGene);
	}

	@Override
	public GWRIndividualFixed recombine(GWRIndividualFixed mother) {
		List<Double> mChromosome = ((GWRIndividualFixed) mother).getChromosome();
		List<Double> nChromosome = new ArrayList<>();

		for (int i = 0; i < chromosome.size(); i++) {
			if (r.nextBoolean())
				nChromosome.add(mChromosome.get(i));
			else
				nChromosome.add(chromosome.get(i));
		}	
		return new GWRIndividualFixed(nChromosome, minGene, maxGene);
	}

	public List<Double> getChromosome() {
		return this.chromosome;
	}
	
	@Override
	public String toString() {
		return "min: " + Collections.min(chromosome) + " " +chromosome.subList(0,Math.min(chromosome.size(),30));
	}

	@Override
	public Map<double[], Double> getSpatialBandwidth(List<double[]> samples, Dist<double[]> gDist ) {
		Map<double[],Double> bandwidth = new HashMap<>();
		for (int i = 0; i < samples.size(); i++) 
			bandwidth.put( samples.get(i), chromosome.get(i) );			
		return bandwidth;
	}
	
	@Override
	public String geneToString(int i) {
		return chromosome.get(i)+"";
	}
}
