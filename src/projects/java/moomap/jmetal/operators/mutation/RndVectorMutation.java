package moomap.jmetal.operators.mutation;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import jmetal.core.Solution;
import jmetal.operators.mutation.Mutation;
import jmetal.util.JMException;

import moomap.jmetal.encodings.variable.TopoMap;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.random.CorrelatedRandomVectorGenerator;
import org.apache.commons.math3.random.GaussianRandomGenerator;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.UncorrelatedRandomVectorGenerator;
import org.apache.commons.math3.stat.descriptive.MultivariateSummaryStatistics;

import spawnn.som.grid.Grid;
import spawnn.som.grid.GridPos;

public class RndVectorMutation extends Mutation {
	
	private static final long serialVersionUID = 1L;
	
	private Double mutationProbability_ = null ;
	private Boolean correlated_ = true;
		
	public RndVectorMutation(HashMap<String, Object> parameters) {
		super(parameters);
		if (parameters.get("probability") != null)
	  		this.mutationProbability_ = (Double)parameters.get("probability") ;
		if (parameters.get("correlated") != null)
	  		this.correlated_ = (Boolean)parameters.get("correlated") ;
	}
		
	@Override
	public Object execute(Object object) throws JMException {
		Solution solution = (Solution)object;
		RandomGenerator rg = new JDKRandomGenerator();
				
		TopoMap tm = ((TopoMap)solution.getDecisionVariables()[0]);
		Grid<double[]> g = tm.grid_;
		for( GridPos rgp : g.getPositions() ) {
			
			if( rg.nextDouble() > mutationProbability_ )
				continue;
												
			// get random vector from multivariate gaussian distribution of neighbors
			List<GridPos> nbs = new ArrayList<GridPos>( );
			nbs.addAll( g.getNeighbours(rgp) );
			nbs.add( rgp );
												
			int pLength = g.getPrototypeAt(rgp).length;
			MultivariateSummaryStatistics mss = new MultivariateSummaryStatistics(pLength, false);
			for( int i = 0; i < nbs.size(); i++ ) 
				mss.addValue(g.getPrototypeAt(nbs.get(i) ) );
						
			GaussianRandomGenerator grg = new GaussianRandomGenerator(rg);	
			RealMatrix cov = mss.getCovariance();
			
			try {
				double[] r;
				
				if( correlated_) {
					CorrelatedRandomVectorGenerator generator  = new CorrelatedRandomVectorGenerator( cov, 1.0e-8 * cov.getNorm(),  grg );
					r = generator.nextVector();
				} else {
					UncorrelatedRandomVectorGenerator generator = new UncorrelatedRandomVectorGenerator(cov.getRowDimension(), grg);
					r = generator.nextVector();
				}
																			
				boolean nan = false;
				for( double d : r ) 
					if( Double.isNaN(d) )
						nan = true;
				
				if( nan ) {
					System.err.println("NAN in random vector, skipping...");
					continue;
				}
				
				double[] orig = g.getPrototypeAt(rgp);
				for( int i = 0; i < orig.length; i++ ) {
					orig[i] += r[i];			
					orig[i] = Math.max( tm.getLowerBound(i), Math.min( tm.getUpperBound(i), orig[i] ) );
				}
				
			} catch( Exception e ) {
				e.printStackTrace();
			}
		}
		return solution;
	}
}
