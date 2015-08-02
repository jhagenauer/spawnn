package moomap.jmetal.problems;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import jmetal.core.Problem;
import jmetal.core.Solution;
import jmetal.util.JMException;

import moomap.jmetal.encodings.solutionType.TopoMapSolutionType;
import moomap.jmetal.encodings.variable.TopoMap;

import org.apache.commons.math3.stat.descriptive.MultivariateSummaryStatistics;

import spawnn.dist.Dist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.DefaultBmuGetter;
import spawnn.som.grid.Grid;
import spawnn.som.grid.GridPos;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;

public class SpatialTopoMapMulti extends Problem {

	private static final long serialVersionUID = 4583308089524086281L;
	
	private List<double[]> samples;
	private BmuGetter<double[]> bg;
	private Dist<double[]> fDist, gDist;
		
	public SpatialTopoMapMulti( int x, int y, List<double[]> samples, Dist<double[]> aDist, Dist<double[]> gDist, Dist<double[]> fDist) throws ClassNotFoundException {
		this.samples = samples;
		this.fDist = fDist;
		this.gDist = gDist;
		
		this.bg = new DefaultBmuGetter<double[]>(aDist);
		
		numberOfVariables_   = 1;
		numberOfObjectives_  = 4;
		numberOfConstraints_ = 0;
		problemName_         = "SpatialTopoMapMulti" ;
		
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
						
		final Grid<double[]> grid = ((TopoMap)solution.getDecisionVariables()[0]).grid_;
		
		Map<GridPos,Set<double[]>> posMapping = SomUtils.getBmuMapping(samples, grid, bg);
		final Map<double[],Set<double[]>> clusters = new HashMap<double[],Set<double[]>>();
		for( GridPos k : posMapping.keySet() )
			clusters.put( grid.getPrototypeAt(k), posMapping.get(k) );
						
		ExecutorService es = Executors.newFixedThreadPool(4); // threads
		List<Future<Map<Integer,Double>>> futures = new ArrayList<Future<Map<Integer,Double>>>();
		
		futures.add( es.submit( new Callable<Map<Integer,Double>>() {
			@Override
			public Map<Integer,Double> call() throws Exception {
				Map<Integer,Double> m = new HashMap<Integer,Double>();
				m.put( 0, DataUtils.getMeanQuantizationError(clusters, fDist) );
				return m;
			}
		}) );
		
		futures.add( es.submit( new Callable<Map<Integer,Double>>() {
			@Override
			public Map<Integer,Double> call() throws Exception {
				Map<Integer,Double> m = new HashMap<Integer,Double>();
				m.put( 1, 1 - SomUtils.getTopoCorrelation(samples, grid, bg, fDist, SomUtils.SPEARMAN_TYPE ) );
				return m;
			}
		}) );
		
		futures.add( es.submit( new Callable<Map<Integer,Double>>() {
			@Override
			public Map<Integer,Double> call() throws Exception {
				Map<Integer,Double> m = new HashMap<Integer,Double>();
				m.put( 2, DataUtils.getMeanQuantizationError(clusters, gDist) );
				return m;
			}
		}) );
		
		futures.add( es.submit( new Callable<Map<Integer,Double>>() {
			@Override
			public Map<Integer,Double> call() throws Exception {
				Map<Integer,Double> m = new HashMap<Integer,Double>();
				m.put( 3, 1 - SomUtils.getTopoCorrelation(samples, grid, bg, gDist, SomUtils.SPEARMAN_TYPE ) );
				return m;
			}
		}) );
		
		es.shutdown();
		
		try {
			for( Future<Map<Integer,Double>> f : futures ) 
				for( Entry<Integer, Double> e : f.get().entrySet() )
					solution.setObjective( e.getKey(), e.getValue() );
		} catch (InterruptedException e) {
			e.printStackTrace();
		} catch (ExecutionException e) {
			e.printStackTrace();
		}
		
		/*solution.setObjective( 0, DataUtils.getQuantError(clusters, fDist)  );
		solution.setObjective( 1, 1 - SomUtils.getTopoCorrelation(samples, grid, bg, fDist, SomUtils.SPEARMAN_TYPE ) );
		solution.setObjective( 2, DataUtils.getQuantError(clusters, gDist) );
		solution.setObjective( 3, 1 - SomUtils.getTopoCorrelation(samples, grid, bg, gDist, SomUtils.SPEARMAN_TYPE ) );*/
	}

}
