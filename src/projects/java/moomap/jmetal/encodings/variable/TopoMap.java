package moomap.jmetal.encodings.variable;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import jmetal.core.Problem;
import jmetal.core.Variable;
import jmetal.util.JMException;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;

public class TopoMap extends Variable {
		
	private static final long serialVersionUID = 1L;
	Problem problem_ ;
	public Grid2D<double[]> grid_;
	
	private double[] lowerBounds_ ;
	private double[] upperBounds_ ;
	
	public TopoMap() {
		this.problem_ = null ;
		this.grid_ = null;
	} 
	
	public TopoMap(int x, int y, int length, Problem problem) {
		this.problem_ = problem ;
		this.grid_ = new Grid2DHex<double[]>(x, y);
		
		//TODO would be nicer, if init with samples
		Random r = new Random();
		for( GridPos p : this.grid_.getPositions() ) {
			double[] d = new double[length];
			for( int i = 0; i < d.length; i++ )
				d[i] = r.nextDouble();
			this.grid_.setPrototypeAt(p, d);
		}
		
		this.lowerBounds_ = new double[length];
		this.upperBounds_ = new double[length];
		for( int i = 0; i < length; i++ ) {
			this.lowerBounds_[i] = problem.getLowerLimit(i);
			this.upperBounds_[i] = problem.getUpperLimit(i);
		}
	}
	
	// deep copy constructor
	public TopoMap(TopoMap tm) {
		problem_ = tm.problem_ ;
		
		int length = tm.grid_.getPrototypes().iterator().next().length;
		
		Map<GridPos,double[]> gm = new HashMap<GridPos,double[]>();
		for( Entry<GridPos, double[]> e : tm.grid_.getGridMap().entrySet() )
			gm.put(e.getKey(), Arrays.copyOf(e.getValue(), length) );
		
		if( tm.grid_ instanceof Grid2DHex )
			grid_ = new Grid2DHex<double[]>( gm );
		else if( tm.grid_ instanceof Grid2D )
			grid_ = new Grid2D<double[]>( gm );
		else
			throw new RuntimeException("Unknown grid type");
		
		this.upperBounds_ = new double[length];
		this.lowerBounds_ = new double[length];
		for( int i = 0; i < length; i++ ) {
			this.lowerBounds_[i] = problem_.getLowerLimit(i);
			this.upperBounds_[i] = problem_.getUpperLimit(i);
		}
	} 

	@Override
	public Variable deepCopy() {
		return new TopoMap(this);
	}
	
	public double getLowerBound(int index) throws JMException {
			return lowerBounds_[index] ;
	} 

	public double getUpperBound(int index) throws JMException {
			return upperBounds_[index];
	} 
	
	@Override 
	public String toString() {
		return grid_.toString();
	}
}
