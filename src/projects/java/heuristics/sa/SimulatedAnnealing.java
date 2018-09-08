package heuristics.sa;

import java.util.Random;

import org.apache.log4j.Logger;

import heuristics.CostCalculator;

public class SimulatedAnnealing {
	
	public static int maxI = 800;
	public static int maxJ = 10;
	public static int maxT = 10;
	
	private static Logger log = Logger.getLogger(SimulatedAnnealing.class);
	
	public static <T extends SAIndividual<T>> T search ( T init, CostCalculator<T> cc ) {
		Random r = new Random();
						
		T x = init;
		double xCost = cc.getCost(x);

		double maxT = 10; 
		int maxI = 100; 
		int maxJ = 10; 
		for( double t = maxT; t > 0.01; t = t*0.9 ) { // 0.001, 0.92
			log.debug(t+","+xCost);
			int j = 0;
			for( int i = 0; i < maxI; ) {
				T y = x.getCopy(); // effectively clone				
				y.step();
				double yCost = cc.getCost(y);
				
				// l muß größer sein, je höher t und je besser y ist.
				// ist y = x, ist l = 1!;
				double l = Math.exp ( ( xCost - yCost )/t);
				if( yCost < xCost || r.nextDouble() < l ) {
					x = y;
					xCost = yCost;
					i++;
					System.out.println(xCost+","+i+","+t);
				} else if( maxJ < ++j ) {
					j = 0;
					i++;
				}
			}
		}
		return x;
	}
}
