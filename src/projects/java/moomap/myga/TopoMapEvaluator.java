package moomap.myga;

import java.util.List;

import spawnn.dist.Dist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.grid.Grid;
import spawnn.som.utils.SomUtils;

public class TopoMapEvaluator implements Evaluator {
	
	List<double[]> samples;
	private BmuGetter<double[]> bg;
	private Dist<double[]> d;
	
	public TopoMapEvaluator( List<double[]> samples, BmuGetter<double[]> bg, Dist<double[]> d ) {
		this.samples = samples;
		this.bg = bg;
		this.d = d;
	}

	@Override
	public double evaluate(GAIndividual i) {
		Grid<double[]> grid = ((TopoMapIndividual)i).grid;
		
		double qe = SomUtils.getMeanQuantError(grid, bg, d, samples);
		//double te = SomUtils.getTopoError(grid, bg, samples);
		double pte = SomUtils.getTopoCorrelation(samples, grid, bg, d, SomUtils.SPEARMAN_TYPE);
		
	    return 2.0 * qe - pte;
	}
}
