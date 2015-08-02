package moomap.myga;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.DefaultBmuGetter;
import spawnn.som.grid.Grid2D;
import spawnn.som.utils.SomUtils;

public class GeneticAlgorithm {
	
	private static Logger log = Logger.getLogger(GeneticAlgorithm.class);
	
	private final static Random r = new Random();
	
	Evaluator evaluator;
			
	public int tournamentSize = 2;
	public double recombProb = 0.9;
	public int threads = 3;
	
	public int evaluations = 0;
				
	public GeneticAlgorithm(Evaluator evaluator) {
		this.evaluator = evaluator;
	}

	public GAIndividual search( List<GAIndividual> init ) {		
		List<GAIndividual> gen = new ArrayList<GAIndividual>(init);
		GAIndividual best = null;
		
		int noImpro = 0;
		int parentSize = init.size();
		int offspringSize = parentSize;
		
		evaluations = init.size();
				
		// init evaluation
		log.debug("init eval ...");
		for( GAIndividual i : init ) 
			i.setValue(evaluator.evaluate(i));
					
		int k = 0;
		log.debug("evolve ...");
		while( noImpro < 100  /*evaluations < 200*/ ) {
						
			// check best and increase noImpro
			noImpro++;
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for( GAIndividual cur : gen ) {
				if( best == null || cur.getValue() < best.getValue()  ) { 
					best = cur;
					noImpro = 0;
					log.debug("found best: "+cur.getValue());
				}
				ds.addValue( cur.getValue() );
			}
			if( noImpro == 0 || k % 100 == 0 || true )				
				log.debug(k+","+evaluations+","+ds.getMin()+","+ds.getMean()+","+ds.getMax()+","+ds.getStandardDeviation() );
															
			// SELECT NEW GEN/POTENTIAL PARENTS
			// elite	
			Collections.sort( gen );	
			List<GAIndividual> elite = new ArrayList<GAIndividual>();
			elite.addAll( gen.subList(0, Math.max( 1, (int)( 0.01*gen.size() ) ) ) );
			for( GAIndividual e : elite ) 
				gen.remove(e);
																						
			List<GAIndividual> selected = new ArrayList<GAIndividual>(elite);
			while( selected.size() < parentSize - elite.size() ) {
				// TOURNAMENT SELECTION
				GAIndividual i = tournament( gen, tournamentSize ); 			
				selected.add( i );
			}		
			gen = selected;	
			gen.addAll(elite);
									
			// GENERATE OFFSPRING
			ExecutorService es = Executors.newFixedThreadPool(threads);
			List<Future<GAIndividual>> offSpring = new ArrayList<Future<GAIndividual>>();
			
			for( int i = 0; i < offspringSize; i++ ) {
				final GAIndividual a = gen.get( r.nextInt( gen.size() ) );
				final GAIndividual b = gen.get( r.nextInt( gen.size() ) );
				
				offSpring.add( es.submit( new Callable<GAIndividual>() {
	
					@Override
					public GAIndividual call() throws Exception {
						GAIndividual child;
						
						if( r.nextDouble() < recombProb )
							child = a.recombine( b );
						else 
							child = a.recombine( a ); // clone
																				
						child.mutate();
						
						child.setValue(evaluator.evaluate(child));
																	
						return child;
					}
				}));	
			}
			es.shutdown();
			
			evaluations += offspringSize;
			
			gen.clear(); // REPLACE GEN WITH OFFSPRING
			for( Future<GAIndividual> f : offSpring ) {
				try {
					gen.add( f.get() );
				} catch (InterruptedException e) {
					e.printStackTrace();
				} catch (ExecutionException e) {
					e.printStackTrace();
				}
			}
			
			k++;
		}			
		return best;
	}
		
	// deterministic tournament selection
	public GAIndividual tournament( List<GAIndividual> gen, int k ) {
		List<GAIndividual> ng = new ArrayList<GAIndividual>();
		
		for( int i = 0; i < k; i++ ) {
			GAIndividual in = gen.get( r.nextInt( gen.size() ) );
			ng.add( in );
		}
		
		Collections.sort( ng );
		return ng.get( 0 ); 		
	}
	
	public GAIndividual binaryProbabilisticTournament( List<GAIndividual> gen, double prob ) {
		Random r = new Random();
		GAIndividual a = gen.get( r.nextInt( gen.size() ) );
		GAIndividual b = gen.get( r.nextInt( gen.size() ) );
		
		if( b.getValue() < a.getValue() ) {
			GAIndividual tmp = a;
			a = b;
			b = tmp;
		}
		if( r.nextDouble() < prob )
			return a;
		else
			return b;
	}
	
	// roulette wheel selection
	public GAIndividual rouletteWheelSelect( List<GAIndividual> gen ) {
		double sum = 0;
		for( GAIndividual in : gen )
			sum += in.getValue();
				
		Random r = new Random();
		double v = r.nextDouble();
		
		double a = 0, b = 0;
		for( int j = 0; j < gen.size(); j++ ) {
			a = b;
			b = (sum - gen.get(j).getValue())/sum + b;
			if( a <= v && v <= b || j+1 == gen.size() && a <= v ) 
				return gen.get(j);
		}
		return null;
	}

	// stochastic universal sampling
	public List<GAIndividual> sus( List<GAIndividual> gen, int n ) {
		List<GAIndividual> l = new ArrayList<GAIndividual>();
		Collections.sort( gen );
		
		double sum = 0;
		for( GAIndividual in : gen )
			sum += in.getValue();

		// intervals
		double ivs[] = new double[gen.size()+1];
		ivs[0] = 0.0f;
		for( int j = 0; j < ivs.length-1; j++ )  
			ivs[j+1] = sum - gen.get(j).getValue() + ivs[j];
		
		double start = r.nextDouble()*sum/n;
		for( int i = 0; i < n; i++ ) {
			double v = start+i*sum/n;
			// binary search of v
			int first = 0, last = ivs.length-1;
			while( true ) {
				int mid = first+(last-first)/2;
				
				if( last - first <= 1 ) {
					l.add( gen.get(mid) );
					break; 
				}
				if( ivs[first] <= v && v <= ivs[mid] ) 
					last = mid;
				else if( ivs[mid] <= v && v <= ivs[last] ) 
					first = mid;
			}
		}
		return l;
	}
	
	public static void main(String[] args) {
		Random r = new Random();
		List<double[]> samples = new ArrayList<double[]>();
		for( int i = 0; i < 250; i++ ) 
			samples.add( new double[]{r.nextDouble(),r.nextDouble()});
		
		final Dist<double[]> dist = new EuclideanDist();
		final BmuGetter<double[]> bg = new DefaultBmuGetter<double[]>(dist);
		
		GeneticAlgorithm ga = new GeneticAlgorithm(new TopoMapEvaluator(samples, bg, dist));
		
		List<GAIndividual> init = new ArrayList<GAIndividual>();
		for( int i = 0; i < 40; i++ )
			init.add( new TopoMapIndividual( 8, 8, 2) );
		
		TopoMapIndividual.mutRate = 1.0/(8*8);
		TopoMapIndividual.recombType = 0;
		ga.recombProb = 0.9;
	
		 GAIndividual best = ga.search(init);
	
		 Grid2D<double[]> grid = ((TopoMapIndividual)best).grid;
		 
		 System.out.println("obj: "+best.getValue());
		 System.out.println("qe: "+SomUtils.getMeanQuantError(grid, bg, dist, samples));
		 System.out.println("te: "+SomUtils.getTopoError(grid, bg, samples));
		 System.out.println("pearson: "+SomUtils.getTopoCorrelation(samples, grid, bg, dist, SomUtils.PEARSON_TYPE) );
		 		    
		 try {
			SomUtils.printGeoGrid( new int[]{0,1}, grid, new FileOutputStream("output/gaTopo.png") );
			SomUtils.printUMatrix(grid, dist, new FileOutputStream("output/gaUMatrix.png") );
			 
			for( int i = 0; i < 2; i++ )
				SomUtils.printComponentPlane(grid, i, new FileOutputStream("output/gaComponent"+i+".png"));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
}
