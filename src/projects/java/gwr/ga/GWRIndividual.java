package gwr.ga;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class GWRIndividual implements GAIndividual<GWRIndividual> {

	protected List<Integer> bw;
	Random r = new Random();

	public GWRIndividual( List<Integer> bw ) {
		this.bw = bw;
	}
	
	@Override
	public GWRIndividual mutate() {
		List<Integer> nBw = new ArrayList<>();
		for( int j = 0; j < bw.size(); j++ ) {
			int h = bw.get(j);
			
			if( r.nextDouble() < 1.0/bw.size() ) {			
				h += (int)Math.round( r.nextGaussian()*4 );
				h = Math.max( h, 6);
			}
			nBw.add(h);
		}
		return new GWRIndividual( nBw );
	}
	
	public List<Integer> getBandwidth() {
		return bw;
	}

	@Override
	public GWRIndividual recombine(GWRIndividual mother) {
		List<Integer> mBw = ((GWRIndividual)mother).getBandwidth();
		List<Integer> nBw = new ArrayList<>();
		for( int i = 0; i < bw.size(); i++)
			if( r.nextBoolean() )
				nBw.add(mBw.get(i));
			else
				nBw.add(bw.get(i));
		return new GWRIndividual(nBw );
	}
		
	public int getBandwidthAt(int i) {
		return this.bw.get(i);
	}
}
