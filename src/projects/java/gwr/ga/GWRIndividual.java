package gwr.ga;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class GWRIndividual extends GAIndividual {

	protected List<Integer> bw;
	
	Random r = new Random();
	CostCalculator<GWRIndividual> cc;

	public GWRIndividual( List<Integer> bw, CostCalculator<GWRIndividual> cc ) {
		this.bw = bw;
		this.cost = getCost();
		this.cc = cc;
	}
	
	@Override
	public GAIndividual mutate() {
		List<Integer> nBw = new ArrayList<>();
		for( int j = 0; j < bw.size(); j++ ) {
			int h = bw.get(j);
			
			if( r.nextDouble() < 1.0/bw.size() ) {
				/*if( h == X.getColumns() || r.nextBoolean() )
					h++;
				else
					h--;*/
				// h = i[r.nextInt(i.length)];
			}
			nBw.add(h);
		}
		return new GWRIndividual( nBw, cc );
	}
	
	public List<Integer> getBandwidth() {
		return bw;
	}

	@Override
	public GAIndividual recombine(GAIndividual mother) {
		List<Integer> mBw = ((GWRIndividual)mother).getBandwidth();
		List<Integer> nBw = new ArrayList<>();
		for( int i = 0; i < bw.size(); i++)
			if( r.nextBoolean() )
				nBw.add(mBw.get(i));
			else
				nBw.add(bw.get(i));
		return new GWRIndividual(nBw, cc );
	}
	
	double cost = Double.NaN;
	
	@Override
	public double getCost() {
		if( Double.isNaN(cost) ) 
			this.cost = cc.getCost(this);
		return cost;
	}
	
	public int getBandwidthAt(int i) {
		return this.bw.get(i);
	}
}
