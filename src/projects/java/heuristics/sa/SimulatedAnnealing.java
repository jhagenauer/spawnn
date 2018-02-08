package heuristics.sa;

import java.util.Random;

import heuristics.CostCalculator;

public class SimulatedAnnealing<T extends SAIndividual<T>> {
	
	CostCalculator<T> eva;
	
	public SimulatedAnnealing(CostCalculator<T> evaluator) {
		this.eva = evaluator;
	}

	public T search( T init ) {
		Random r = new Random();
						
		T x = init;
		double xCost = eva.getCost(x);

		double maxT = 10; 
		int maxI = 800; 
		int maxJ = 10; 
		for( double t = maxT; t > 0.001; t = t*0.92 ) { // 0.001, 0.92
			//log.debug(t+","+x.getValue());
			int j = 0;
			for( int i = 0; i < maxI; ) {
				T y = x.getCopy(); // effectively clone				
				y.step();
				double yCost = eva.getCost(y);
				
				// l muß größer sein, je höher t und je besser y ist.
				// ist y = x, ist l = 1!;
				double l = Math.exp ( ( xCost - yCost )/t);
				if( yCost < xCost || r.nextDouble() < l ) {
					x = y;
					xCost = yCost;
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
