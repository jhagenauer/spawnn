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

	public GWRIndividual( List<Double> chromosome, double sd ) {
		this.chromosome = chromosome;
		this.sd = sd;
	}
	
	public static boolean useNB4Mut = false;
	public static Map<Integer, Set<Integer>> cmI;
		
	@Override
	public GWRIndividual mutate() {
		List<Double> nBw = new ArrayList<>();
		for( int j = 0; j < chromosome.size(); j++ ) {
			double h = chromosome.get(j);
			if( r.nextDouble() < 1.0/chromosome.size() ) {	
				if( !useNB4Mut ) {
					h += r.nextGaussian()*sd;
				} else {
					SummaryStatistics ds = new SummaryStatistics();
					for( int i : cmI.get(j) )
						ds.addValue( chromosome.get(i) );
					h += r.nextGaussian()*ds.getStandardDeviation()*sd;
				}
			}
			nBw.add(h);
		}
		return new GWRIndividual( nBw, sd );
	}
	
	public static boolean meanRecomb = false;
			
	@Override
	public GWRIndividual recombine(GWRIndividual mother) {
		List<Double> mBw = ((GWRIndividual)mother).getChromosome();
		List<Double> nBw = new ArrayList<>();
		
		for( int i = 0; i < chromosome.size(); i++)
			if( !meanRecomb ) {
				if( r.nextBoolean() ) 
					nBw.add(mBw.get(i));
				else
					nBw.add(chromosome.get(i));
			} else {
				nBw.add( (chromosome.get(i)+mBw.get(i))/2 );
			}
		return new GWRIndividual( nBw, sd );
	}
	
	public List<Double> getChromosome() {
		return this.chromosome;
	}
	
	public double getBandwidthAt(int i) {
		return chromosome.get(i);
	}
}
