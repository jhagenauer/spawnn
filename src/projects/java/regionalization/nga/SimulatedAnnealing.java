package regionalization.nga;

import java.util.Random;

import org.apache.log4j.Logger;

import myga.Evaluator;
import myga.GAIndividual;

public class SimulatedAnnealing<T extends GAIndividual<T>> {
	private static Logger log = Logger.getLogger(SimulatedAnnealing.class);
	
	Evaluator<T> eva;
	
	public SimulatedAnnealing(Evaluator<T> evaluator) {
		this.eva = evaluator;
	}

	public T search( T init ) {
		Random r = new Random();
						
		T x = init;
		x.setValue( eva.evaluate(x) );

		double maxT = 10; 
		int maxI = 1200; 
		int maxJ = 10; 
		for( double t = maxT; t > 0.001; t = t*0.92 ) { // 0.001, 0.92
			int j = 0;
			for( int i = 0; i < maxI; ) {
				T y = x.recombine(x); // effectively clone
				y.mutate();
				y.setValue( eva.evaluate(y) );
				
				// l muß größer sein, je höher t und je besser y ist.
				// ist y = x, ist l = 1!;
				double l = Math.exp ( ( x.getValue() - y.getValue() )/t);
				if( y.getValue() < x.getValue() || r.nextDouble() < l ) {
					x = y;
					i++;
				} else if( maxJ < ++j ) {
					j = 0;
					i++;
				}
			}
		}
		return x;
	}
}
