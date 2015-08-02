package moomap.jmetal.problems;

import java.util.List;

import moomap.jmetal.encodings.solutionType.TopoMapSolutionType;
import moomap.jmetal.encodings.variable.TopoMap;

import org.apache.commons.math3.stat.descriptive.MultivariateSummaryStatistics;

import jmetal.core.Problem;
import jmetal.core.Solution;
import jmetal.util.JMException;
import spawnn.dist.Dist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.grid.Grid;
import spawnn.som.utils.SomUtils;

public class TopoMapMulti extends Problem {

	private static final long serialVersionUID = 4583308089564086281L;
	
	private List<double[]> samples;
	private BmuGetter<double[]> bg;
	private Dist<double[]> dist;
		
	public TopoMapMulti( int x, int y, List<double[]> samples, BmuGetter<double[]> bg, Dist<double[]> dist ) throws ClassNotFoundException {
		this.samples = samples;
		this.bg = bg;
		this.dist = dist;
		
		numberOfVariables_   = 1;
		numberOfObjectives_  = 2;
		numberOfConstraints_ = 0;
		problemName_         = "TopoMapMulti" ;
		
		MultivariateSummaryStatistics mss = new MultivariateSummaryStatistics(samples.get(0).length, false);
		for( double[] d : samples ) 
			mss.addValue( d );
		lowerLimit_ = mss.getMin();
		upperLimit_ = mss.getMax();

		solutionType_ = new TopoMapSolutionType( x, y, samples.get(0).length, this );
	}

	@Override
	public void evaluate(Solution solution) throws JMException {
		if( !(solution.getType() instanceof TopoMapSolutionType) )
			throw new RuntimeException("SolutionType "+solution.getType()+" not valid.");
						
		Grid<double[]> grid = ((TopoMap)solution.getDecisionVariables()[0]).grid_;
		
		double qe = SomUtils.getMeanQuantError(grid, bg, dist, samples);
		//double te = SomUtils.getTopoError(grid, bg, samples);
		double pte = SomUtils.getTopoCorrelation(samples, grid, bg, dist, SomUtils.SPEARMAN_TYPE);
		
	    solution.setObjective( 0, qe );
		solution.setObjective( 1, 1-pte );
	}

}
