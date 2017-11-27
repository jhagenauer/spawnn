package gwr.ga;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

public class GWRIndividual implements GAIndividual<GWRIndividual> {

	protected Random r = new Random();
	
	protected List<Double> bw;
	protected double sd;

	public GWRIndividual( List<Double> bw, double sd ) {
		this.bw = bw;
		this.sd = sd;
	}
	
	public static boolean useNB4Mut = false;
	public static Map<Integer, Set<Integer>> cmI;
		
	@Override
	public GWRIndividual mutate() {
		List<Double> nBw = new ArrayList<>();
		for( int j = 0; j < bw.size(); j++ ) {
			double h = bw.get(j);
			if( r.nextDouble() < 1.0/bw.size() ) {	
				if( !useNB4Mut ) {
					h += r.nextGaussian()*sd;
				} else {
					DescriptiveStatistics ds = new DescriptiveStatistics();
					for( int i : cmI.get(j) )
						ds.addValue( bw.get(i) );
					h += (int)Math.round( r.nextGaussian()*ds.getStandardDeviation()*sd );
				}
			}
			nBw.add(h);
		}
		return new GWRIndividual( nBw, sd );
	}
	
	public static boolean meanRecomb = false;
			
	@Override
	public GWRIndividual recombine(GWRIndividual mother) {
		List<Double> mBw = ((GWRIndividual)mother).getBandwidth();
		List<Double> nBw = new ArrayList<>();
		
		for( int i = 0; i < bw.size(); i++)
			if( !meanRecomb ) {
				if( r.nextBoolean() ) 
					nBw.add(mBw.get(i));
				else
					nBw.add(bw.get(i));
			} else {
				nBw.add( (bw.get(i)+mBw.get(i))/2 );
			}
		return new GWRIndividual( nBw, sd );
	}
	
	public List<Double> getBandwidth() {
		return this.bw;
	}
}
