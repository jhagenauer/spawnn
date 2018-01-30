package gwr.ga;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

public class GWRIndividual implements GAIndividual<GWRIndividual> {

	protected Random r = new Random();
	
	protected List<Double> chromosome;
	protected double sd;
	protected double minGene, maxGene;

	public GWRIndividual( List<Double> chromosome, double sd, double maxGene, double minGene ) {
		this.chromosome = chromosome;
		for( int i = 0; i < this.chromosome.size(); i++ ) {
			double h = this.chromosome.get(i);
			h = Math.max( minGene, Math.min( maxGene, h) );
			this.chromosome.set(i, h);
		}
		this.sd = sd;
		this.minGene = minGene;
		this.maxGene = maxGene;
	}
	
	public static boolean mutationMode = false;
	public static Map<Integer, Set<Integer>> cmI;
		
	@Override
	public GWRIndividual mutate() {
		List<Double> nBw = new ArrayList<>();
		for( int j = 0; j < chromosome.size(); j++ ) {
			double h = chromosome.get(j);
			if( r.nextDouble() < 1.0/chromosome.size() ) {	
				if( !mutationMode ) {
					h += r.nextGaussian()*sd;
				} else {
					SummaryStatistics ds = new SummaryStatistics();
					for( int i : cmI.get(j) )
						ds.addValue( chromosome.get(i) );
					h += r.nextGaussian()*ds.getStandardDeviation()*sd;
				}
				h = Math.max( minGene, Math.min( maxGene, h) );
			}
			nBw.add(h );
		}
		return new GWRIndividual( nBw, sd, maxGene, minGene );
	}
	
	public static boolean meanRecomb = false;
			
	@Override
	public GWRIndividual recombine(GWRIndividual mother) {
		List<Double> mChromosome = ((GWRIndividual)mother).getChromosome();
		List<Double> nChromosome = new ArrayList<>();
		
		for( int i = 0; i < chromosome.size(); i++)
			if( !meanRecomb ) {
				if( r.nextBoolean() ) 
					nChromosome.add( mChromosome.get(i) );
				else
					nChromosome.add(chromosome.get(i));
			} else {
				nChromosome.add( (chromosome.get(i)+mChromosome.get(i))/2 );
			}
		return new GWRIndividual( nChromosome, sd, maxGene, minGene );
	}
	
	public List<Double> getChromosome() {
		return this.chromosome;
	}
	
	public double getGeneAt(int i) {
		return chromosome.get(i);
	}
}
