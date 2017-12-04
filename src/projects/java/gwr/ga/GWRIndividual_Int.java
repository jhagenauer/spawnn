package gwr.ga;

import java.util.ArrayList;
import java.util.List;

public class GWRIndividual_Int extends GWRIndividual {

	public GWRIndividual_Int(List<Double> chromosome, double sd) {
		super(chromosome, sd);
	}
	
	private int getPoissonRandom(double mean) {
	    double L = Math.exp(-mean);
	    int k = 0;
	    double p = 1.0;
	    do {
	        p = p * r.nextDouble();
	        k++;
	    } while (p > L);
	    return k - 1;
	}
	
	@Override
	public GWRIndividual mutate() {
		List<Double> nBw = new ArrayList<>();
		for( int j = 0; j < chromosome.size(); j++ ) {
			double h = chromosome.get(j);
			if( r.nextDouble() < 1.0/chromosome.size() ) {
				double mean = 0;
				for( int i : cmI.get(j) )
					mean += chromosome.get(i);
				mean /= cmI.get(j).size();
				h = getPoissonRandom(mean);
			}
			nBw.add(h);
		}
		return new GWRIndividual_Int( nBw, sd );
	}
				
	@Override
	public GWRIndividual recombine(GWRIndividual mother) {
		List<Double> mBw = ((GWRIndividual)mother).getChromosome();
		List<Double> nBw = new ArrayList<>();
		for( int i = 0; i < chromosome.size(); i++)
			if( r.nextBoolean() ) 
				nBw.add(mBw.get(i));
			else
				nBw.add(chromosome.get(i));
		return new GWRIndividual_Int( nBw, sd );
	}
}
