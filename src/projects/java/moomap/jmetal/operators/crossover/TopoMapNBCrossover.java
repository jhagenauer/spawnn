package moomap.jmetal.operators.crossover;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import moomap.jmetal.encodings.variable.TopoMap;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;

import jmetal.core.Solution;
import jmetal.operators.crossover.Crossover;
import jmetal.util.JMException;
import jmetal.util.PseudoRandom;
import spawnn.som.grid.Grid;
import spawnn.som.grid.GridPos;

public class TopoMapNBCrossover extends Crossover {

	private Double crossoverProbability_ = null;
	
	public static int recombType = 0; 
	
	public TopoMapNBCrossover(HashMap<String, Object> parameters) {
		super(parameters);
		if (parameters.get("probability") != null)
			crossoverProbability_ = (Double) parameters.get("probability");
	}

	private static final long serialVersionUID = 1L;

	@Override
	public Object execute(Object object) throws JMException {
		RandomGenerator rg = new JDKRandomGenerator();
		
		Solution[] parents = (Solution[]) object;
		Solution[] offspring = new Solution[] { new Solution(parents[0]), new Solution(parents[1]) };
		
		if ( rg.nextDouble() < crossoverProbability_) {
			
			Grid<double[]> g0 = ((TopoMap) offspring[0].getDecisionVariables()[0]).grid_;
			Grid<double[]> g1 = ((TopoMap) offspring[1].getDecisionVariables()[0]).grid_;

			// get random grid pos
			List<GridPos> gps = new ArrayList<GridPos>(g0.getPositions());
			GridPos rgp = gps.get(PseudoRandom.randInt(0, gps.size() - 1));
			
			Map<Integer,Set<GridPos>> distMap = new HashMap<Integer,Set<GridPos>>();
			for( GridPos p : g0.getPositions() ) {
				int d = g0.dist(rgp, p);
				if( !distMap.containsKey(d) )
					distMap.put( d, new HashSet<GridPos>() );
				distMap.get(d).add(p);
			}
			List<Integer> sortedKeys = new ArrayList<Integer>(distMap.keySet());
			Collections.sort(sortedKeys);
					
			Set<GridPos> exchangeSet = new HashSet<GridPos>();				
			/* statt radius, wäre es nicht nett, einen cluster (z.B. mit ward erstellt) auszutauschen?
			 * was wäre der vorteil? ->
			 */
			if( recombType == 0 ) { 
				// nearest n
				int n = rg.nextInt( g0.getPositions().size() );
				
				for( int i : sortedKeys )
					for( GridPos p : distMap.get(i) )
						if( exchangeSet.size() <= n )
							exchangeSet.add(p);
									
			} else if( recombType == 1 ) {
				// random radius
				int r = rg.nextInt(sortedKeys.size());
				
				for( int i : sortedKeys )
					if( i <= r )
						exchangeSet.addAll( distMap.get(i) );
						
			} else if( recombType == 2 ){
				// 1
				exchangeSet.addAll( distMap.get(0) );
				exchangeSet.addAll( distMap.get(1) );
			} else if( recombType == 3 ) {
				int n = g0.size()/2;
				
				for( int i : sortedKeys )
					for( GridPos p : distMap.get(i) )
						if( exchangeSet.size() <= n )
							exchangeSet.add(p);
				
			} else if( recombType == 4 ) {
				int r = sortedKeys.size()/2;
				
				for( int i : sortedKeys )
					if( i <= r )
						exchangeSet.addAll( distMap.get(i) );
			} 
			
			for( GridPos p : exchangeSet ) {
				double[] old = g1.setPrototypeAt(p, g0.getPrototypeAt(p) );
				g0.setPrototypeAt( p, old);
			}
		}		
		return offspring;
	}

}
