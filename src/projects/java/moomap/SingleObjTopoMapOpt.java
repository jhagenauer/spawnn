package moomap;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.logging.FileHandler;
import java.util.logging.Logger;

import jmetal.core.Algorithm;
import jmetal.core.Operator;
import jmetal.core.Problem;
import jmetal.core.Solution;
import jmetal.core.SolutionSet;
import jmetal.metaheuristics.singleObjective.geneticAlgorithm.gGA;
import jmetal.operators.selection.BinaryTournament;
import jmetal.util.Configuration;

import moomap.jmetal.encodings.variable.TopoMap;
import moomap.jmetal.operators.crossover.TopoMapNBCrossover;
import moomap.jmetal.operators.mutation.RndVectorMutation;
import moomap.jmetal.problems.TopoMapSingle;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.DefaultBmuGetter;
import spawnn.som.grid.Grid2D;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;

public class SingleObjTopoMapOpt {
	public static Logger log;
	public static FileHandler fileHandler_;

	/*public static void main( String[] args ) {
		try {
			int xDim = 8, yDim = 8;
			
			Random r = new Random();
			List<double[]> samples = new ArrayList<double[]>();
			for( int i = 0; i < 250; i++ ) 
				samples.add( new double[]{r.nextDouble(),r.nextDouble()});
			
			final Dist<double[]> dist = new EuclideanDist();
			final BmuGetter<double[]> bg = new DefaultBmuGetter<double[]>(dist);
			
			log = Configuration.logger_ ;
		    fileHandler_ = new FileHandler("moosom.log"); 
		    log.addHandler(fileHandler_) ;
		
			Problem problem =  new TopoMapSingle( xDim, yDim, samples, bg, dist );
			Algorithm algorithm = new gGA(problem);
						
			algorithm.setInputParameter("populationSize",40);
			algorithm.setInputParameter("maxEvaluations",50000);
	
		    HashMap<String, Object> parameters = new HashMap<String,Object>() ;
		    parameters.put("probability", 0.9) ; // 0.9
		    Operator crossover = new TopoMapNBCrossover(parameters);
		   
		    TopoMapNBCrossover.recombType = 3;
		    
		    parameters = new HashMap<String,Object>() ;
		    parameters.put("probability", 1.0/(xDim*yDim) ) ;
		    Operator mutation = new TopoMapMutation(parameters);   
		    
		    TopoMapMutation.mutType = 0;
		    SomUtils.topoType = 1;
		    
		    Operator selection = new BinaryTournament(null) ;                           
		    algorithm.addOperator("crossover",crossover);
		    algorithm.addOperator("mutation",mutation);
		    algorithm.addOperator("selection",selection);

		    long initTime = System.currentTimeMillis();
		    SolutionSet population = algorithm.execute();
		    long estimatedTime = System.currentTimeMillis() - initTime;
		
		    log.info("Total execution time: "+estimatedTime + "ms");

		    population.printVariablesToFile("VAR");    
		    population.printObjectivesToFile("FUN");
		    
		    log.info("first: "+population.get(0).getObjective(0));
		    Solution best = null;
		    for( int i = 0; i < population.size(); i++ ) {	
		    	if( best == null || population.get(i).getObjective(0) < best.getObjective(0) )
		    		best = population.get(i);
		    }
		    
		    log.info("Best: "+best.getObjective(0));
		    Grid2D<double[]> grid = (Grid2D<double[]>)((TopoMap)best.getDecisionVariables()[0]).grid_;
		    log.info("qe: "+SomUtils.getQuantError(grid, bg, dist, samples));
		    log.info("te: "+SomUtils.getTopoError(grid, bg, samples));
		    log.info("pearson: "+SomUtils.getTopoCorrelation(samples, grid, bg, dist) );
		    
		    SomUtils.printTopologyGeo( new int[]{0,1}, grid, new FileOutputStream("output/gaTopo.png") );
		    SomUtils.printUMatrix(grid, dist, new FileOutputStream("output/gaUMatrix.png") );
		    
		    for( int i = 0; i < 2; i++ )
				SomUtils.printComponentPlanes(grid, i, new FileOutputStream("output/gaComponent"+i+".png"));
		    		  		  	
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		} catch (SecurityException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (JMException e) {
			e.printStackTrace();
		}	
	}*/
	
	public static void main( String[] args ) {
		try {
			final int xDim = 8, yDim = 8;
			
			final List<double[]> samples = DataUtils.readCSV(new FileInputStream("data/iris.csv") );
			DataUtils.normalize(samples);
						
			final Dist<double[]> dist = new EuclideanDist();
			final BmuGetter<double[]> bg = new DefaultBmuGetter<double[]>(dist);
			
			log = Configuration.logger_ ;
		    fileHandler_ = new FileHandler("moosom.log"); 
		    log.addHandler(fileHandler_) ;
		    
		   
		    int[][] settings = new int[][]{ {0,0,1},{3,0,1},{4,0,1} };
		    
		    for( int[] setting : settings ) {
		    	final int crossType = setting[0];
		    	final int mutType = setting[1];
		    	final int topoType = setting[2];                     
		    
		    // zu testen: topo-type, mutation, crossover
			//for( final int crossType : new int[]{ 0, 1, 2, 3, 4 } ) {
			//	for( final int mutType : new int[]{ 0, 1 } ) {
			//	    for( final int topoType : new int[]{ 0, 1 } ) {
				    	
				    	ExecutorService es = Executors.newFixedThreadPool(16);
						List<Future<double[]>> futures = new ArrayList<Future<double[]>>();
				    	log.info("crossType: "+crossType+", mutType: "+mutType+", topType: "+topoType );
				    				    	
				    	for( int j = 0; j < 100; j++ ) {
				    		
				    		futures.add( es.submit( new Callable<double[]>() {
				    			
								@Override
								public double[] call() throws Exception {
									
									Problem problem =  new TopoMapSingle( xDim, yDim, samples, bg, dist );
									Algorithm algorithm = new gGA(problem);
												
									algorithm.setInputParameter("populationSize",40);
									algorithm.setInputParameter("maxEvaluations",50000);
							
								    HashMap<String, Object> parameters = new HashMap<String,Object>() ;
								    parameters.put("probability", 0.9) ; // 0.9
								    Operator crossover = new TopoMapNBCrossover(parameters);
								    TopoMapNBCrossover.recombType = crossType;
								    				   
								    parameters = new HashMap<String,Object>() ;
								    parameters.put("probability", 1.0/(xDim*yDim) ) ;
								    Operator mutation = new RndVectorMutation(parameters);       
								   						
								    Operator selection = new BinaryTournament(null) ;                           
								    algorithm.addOperator("crossover",crossover);
								    algorithm.addOperator("mutation",mutation);
								    algorithm.addOperator("selection",selection);
						
								    SolutionSet population = algorithm.execute();
								    		    
								    Solution best = null;
								    for( int i = 0; i < population.size(); i++ ) {	
								    	if( best == null || population.get(i).getObjective(0) < best.getObjective(0) )
								    		best = population.get(i);
								    }
								    
								    Grid2D<double[]> grid = (Grid2D<double[]>)((TopoMap)best.getDecisionVariables()[0]).grid_;
								    
								    return new double[]{
								    		best.getObjective(0),
								    	    SomUtils.getMeanQuantError(grid, bg, dist, samples),
								    	    SomUtils.getTopoError(grid, bg, samples),
								    	    SomUtils.getTopoCorrelation(samples, grid, bg, dist,SomUtils.SPEARMAN_TYPE)
								    };
								}
				    		}));				    		
					    }
				    	
				    	es.shutdown();
				    	
				    	DescriptiveStatistics qe = new DescriptiveStatistics();
				    	DescriptiveStatistics te = new DescriptiveStatistics();
				    	DescriptiveStatistics cor = new DescriptiveStatistics();
				    	DescriptiveStatistics obj = new DescriptiveStatistics();
						
				    	for( Future<double[]> f : futures ) {
							try {
								double[] d = f.get();
								obj.addValue( d[0] );
								qe.addValue( d[1] );
								te.addValue( d[2] );
								cor.addValue( d[3] );
							} catch (InterruptedException e) {
								e.printStackTrace();
							} catch (ExecutionException e) {
								e.printStackTrace();
							}
						}
				    	
				    	log.info("obj: "+obj.getMin()+","+obj.getMean()+","+obj.getMax()+","+obj.getStandardDeviation() );
				    	log.info("qe : "+qe.getMin()+","+qe.getMean()+","+qe.getMax()+","+qe.getStandardDeviation() );
				    	log.info("te : "+te.getMin()+","+te.getMean()+","+te.getMax()+","+te.getStandardDeviation() );
				    	log.info("cor: "+cor.getMin()+","+cor.getMean()+","+cor.getMax()+","+cor.getStandardDeviation() );
				    	
				    }
		//		}
		//	}
		    		  		  	
		} catch (SecurityException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}	
	}
}
