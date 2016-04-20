package houseprice.optimSubMarkets;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import myga.GAIndividual;

// genetic operators maintain constant sum of ones
public class MaintainSumIndividual extends GAIndividual {
	
	private boolean[] a;
	private Random r = new Random();
	
	public MaintainSumIndividual(boolean[] a) {
		this.a = a;
	}

	@Override
	public void mutate() {
		// set random false to true
		while( true ) {
			int idx = r.nextInt(a.length);
			if( a[idx] ) {
				a[idx] = false;
				break;
			}
		}		
				
		// set random false to true
		while( true ) {
			int idx = r.nextInt(a.length);
			if( !a[idx] ) {
				a[idx] = true;
				break;
			}
		}	
	}

	@Override
	public GAIndividual recombine(GAIndividual mother) {
		int ca = 0, cb = 0;
		boolean[] b = ((MaintainSumIndividual)mother).getChromosome();
		List<Integer> p = new ArrayList<Integer>();
		for( int i = 0; i < a.length; i++ ) {
			if( a[i] )
				ca++;
			if( b[i] )
				cb++;
			if( ca == cb )
				p.add(i);
		}
		int idx = p.get(r.nextInt(p.size()));
		boolean[] c = Arrays.copyOf(a, a.length);
		for( int i = 0; i <= idx; i++ )
			c[i] = b[i];
		MaintainSumIndividual child = new MaintainSumIndividual(c);
		return child;
	}
	
	public boolean[] getChromosome() {
		return a;
	}
}
