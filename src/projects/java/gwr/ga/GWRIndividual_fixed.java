package gwr.ga;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import ga.GAIndividual;
import heuristics.sa.SAIndividual;

public class GWRIndividual_fixed implements GAIndividual<GWRIndividual_fixed>, SAIndividual<GWRIndividual_fixed> {

	protected Random r;

	protected List<Integer> chromosome;
	protected int minGene, maxGene;
	
	public static Map<Integer, Set<Integer>> cmI;

	public static double sd;

	public GWRIndividual_fixed(List<Integer> chromosome, int minGene, int maxGene) {
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
	public GWRIndividual_fixed mutate() {
		List<Integer> nChromosome = new ArrayList<>();
		for (int j = 0; j < chromosome.size(); j++) {
			int h = chromosome.get(j);

			if (r.nextDouble() < 1.0 / chromosome.size()) {
											
				double d = r.nextGaussian()*sd;
				if( d < 0 )
					h += (int)Math.floor(d);
				else
					h += (int)Math.ceil(d);
								
				// h += r.nextInt(25)-12;			

				h = Math.max(minGene, Math.min(maxGene, h));
			}
			nChromosome.add(h);
		}
		
		GWRIndividual_fixed i = new GWRIndividual_fixed(nChromosome, minGene, maxGene);
		return i;
	}

	@Override
	public GWRIndividual_fixed recombine(GWRIndividual_fixed mother) {
		List<Integer> mChromosome = ((GWRIndividual_fixed) mother).getChromosome();
		List<Integer> nChromosome = new ArrayList<>();

		for (int i = 0; i < chromosome.size(); i++) {
			if (r.nextBoolean())
				nChromosome.add(mChromosome.get(i));
			else
				nChromosome.add(chromosome.get(i));
		}
		
		GWRIndividual_fixed i = new GWRIndividual_fixed(nChromosome, minGene, maxGene);		
		return i;
	}

	@Override
	public GWRIndividual_fixed getCopy() {
		return new GWRIndividual_fixed(chromosome, minGene, maxGene);
	}

	public List<Integer> getChromosome() {
		return this.chromosome;
	}

	public int getGeneAt(int i) {
		return chromosome.get(i);
	}

	@Override
	public void step() {
		for (int j = 0; j < chromosome.size(); j++) {
			int h = chromosome.get(j);

			if (r.nextDouble() < 1.0 / chromosome.size()) {
								
				if (r.nextBoolean())
					h = (int) (h + 1);
				else
					h = (int) (h - 1);

				h = Math.max(minGene, Math.min(maxGene, h));
			}
			chromosome.add(h);
		}
	}
	
	@Override
	public String toString() {
		return "min: " + Collections.min(chromosome) + " " +chromosome.subList(0,Math.min(chromosome.size(),30));
	}
}
