package moomap.myga;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.random.CorrelatedRandomVectorGenerator;
import org.apache.commons.math3.random.GaussianRandomGenerator;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.RandomVectorGenerator;
import org.apache.commons.math3.stat.descriptive.MultivariateSummaryStatistics;
import org.apache.log4j.Logger;

import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;

public class TopoMapIndividual implements GAIndividual {
	
	private static Logger log = Logger.getLogger(TopoMapIndividual.class);
	
	public static double mutRate;
	
	public Grid2D<double[]> grid;
	public double value = 0.0;
	
	public static int recombType = 0;
	
	private RandomGenerator rg = new JDKRandomGenerator();
	
	public TopoMapIndividual(Grid2D<double[]> grid ) {
		this.grid = grid;	
	}
	
	public TopoMapIndividual(int x, int y, int length) {
		this.grid = new Grid2DHex<double[]>(x, y);
		
		//TODO would be nicer, if init with samples
		Random r = new Random();
		for( GridPos p : this.grid.getPositions() ) {
			double[] d = new double[length];
			for( int i = 0; i < d.length; i++ )
				d[i] = r.nextDouble();
			this.grid.setPrototypeAt(p, d);
		}
	}

	@Override
	public void mutate() {
		for( GridPos p : grid.getPositions() ) {
		
			if( rg.nextDouble() > mutRate )
				continue;
						
			// add random vector from multivariate gaussian distribution of neighbors
			List<GridPos> nbs = new ArrayList<GridPos>();
			nbs.addAll( grid.getNeighbours(p));
			nbs.add(p);
			/*for( GridPos p1 : grid.getPositions() )
				if( grid.dist(p, p1) <= 1)
					nbs.add(p1);*/
						
			int pLength = grid.getPrototypeAt(p).length;
			MultivariateSummaryStatistics mss = new MultivariateSummaryStatistics(pLength, false);
			for( int i = 0; i < nbs.size(); i++ ) 
				mss.addValue(grid.getPrototypeAt(nbs.get(i) ) );
						
			GaussianRandomGenerator grg = new GaussianRandomGenerator(rg);	
			try {
				// corrleated, add
				RealMatrix cov = mss.getCovariance();
				RandomVectorGenerator generator  = new CorrelatedRandomVectorGenerator( new double[pLength], cov, 1.0e-12 * cov.getNorm(),  grg);
				double[] r = generator.nextVector();
					
				boolean nan = false;
				for( double d : r ) 
					if( Double.isNaN(d) )
						nan = true;
				
				if( nan ) {
					log.warn("NAN in random vector, skipping...");
					continue;
				}
												
				double[] orig = grid.getPrototypeAt(p);
				for( int i = 0; i < orig.length; i++ ) {
					orig[i] += r[i];
					orig[i] = Math.max(0.0, Math.min( 1.0, orig[i] ) ); // clip to bounds
				}
				
			} catch( Exception e ) {
				e.printStackTrace();
			}
		}
	}

	@Override
	public GAIndividual recombine(GAIndividual mother) {	
		Random rnd = new Random();
		
		Grid2D<double[]> ng = new Grid2DHex<double[]>(grid.getSizeOfDim(0),grid.getSizeOfDim(1));
		Grid2D<double[]> motherGrid = ((TopoMapIndividual)mother).grid;
		
		// get random grid pos
		List<GridPos> gps = new ArrayList<GridPos>(ng.getPositions());
		GridPos rgp = gps.get(rnd.nextInt(gps.size()));
		
		Map<Integer,Set<GridPos>> distMap = new HashMap<Integer,Set<GridPos>>();
		for( GridPos p2 : grid.getPositions() ) {
			int d = grid.dist(rgp, p2);
			if( !distMap.containsKey(d) )
				distMap.put( d, new HashSet<GridPos>() );
			distMap.get(d).add(p2);
		}
		List<Integer> sortedKeys = new ArrayList<Integer>(distMap.keySet());
		Collections.sort(sortedKeys);
				
		Set<GridPos> exchangeSet = new HashSet<GridPos>();
			
		if( recombType == 0 ) {
			// nearest n
			int n = rg.nextInt( grid.getPositions().size() );
			
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
			int n = grid.size()/2;
			
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
		
		// copy from Mother
		for( GridPos p2 : grid.getPositions() ) {
			double[] d;
			if( exchangeSet.contains(p2) )
				d = motherGrid.getPrototypeAt(p2);
			else
				d = grid.getPrototypeAt(p2);
			ng.setPrototypeAt(p2, Arrays.copyOf(d, d.length));
		}
					
		return new TopoMapIndividual(ng);
	}

	@Override
	public void setValue(double value) {
		this.value = value;
	}

	@Override
	public double getValue() {
		return value;
	}
	
	@Override
	public int compareTo(GAIndividual i) {
		return Double.compare(getValue(), i.getValue());
	}
}
