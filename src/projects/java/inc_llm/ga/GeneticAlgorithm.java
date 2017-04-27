package inc_llm.ga;

import java.io.File;
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
import org.jblas.DoubleMatrix;

import com.vividsolutions.jts.geom.Point;

import chowClustering.LinearModel;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

public class GeneticAlgorithm {
	
	private static Logger log = Logger.getLogger(GeneticAlgorithm.class);
	private final static Random r = new Random();
	int threads = Math.max(1 , Runtime.getRuntime().availableProcessors() -1 );;
	
	public int tournamentSize = 2;
	public double recombProb = 0.7;
				
	public GAIndividual search( List<GAIndividual> init ) {		
		List<GAIndividual> gen = new ArrayList<GAIndividual>(init);
		GAIndividual best = null;
		
		int noImpro = 0;
		int parentSize = init.size();
		int offspringSize = parentSize*2;
				
		int maxK = 1000;
		int k = 0;
		while( /*k < maxK /*||*/ noImpro < 100 ) {
				
			// check best and increase noImpro
			noImpro++;
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for( GAIndividual cur : gen ) {
				if( best == null || cur.getCost() < best.getCost()  ) { 
					best = cur;
					noImpro = 0;
					log.info("best "+cur.getCost()+" "+cur);
				}
				ds.addValue( cur.getCost() );
			}
			if( noImpro == 0 || k % 100 == 0 )
				log.info(k+","+ds.getMin()+","+ds.getMean()+","+ds.getMax()+","+ds.getStandardDeviation() );
																		
			// SELECT NEW GEN/POTENTIAL PARENTS
			// elite
			Collections.sort( gen );	
			List<GAIndividual> elite = new ArrayList<GAIndividual>();
			//elite.addAll( gen.subList(0, Math.max( 1, (int)( 0.01*gen.size() ) ) ) );
			gen.removeAll(elite);
												
			List<GAIndividual> selected = new ArrayList<GAIndividual>(elite);
			while( selected.size() < parentSize ) {
				GAIndividual i = tournament( gen, tournamentSize );			
				selected.add( i );
			}		
			gen = selected;	
			
			// GENERATE OFFSPRING
			ExecutorService es = Executors.newFixedThreadPool(threads);
			List<Future<GAIndividual>> futures = new ArrayList<Future<GAIndividual>>();
			
			for( int i = 0; i < offspringSize; i++ ) {
				final GAIndividual a = gen.get( r.nextInt( gen.size() ) );
				final GAIndividual b = gen.get( r.nextInt( gen.size() ) );
				
				futures.add( es.submit( new Callable<GAIndividual>() {
	
					@Override
					public GAIndividual call() throws Exception {
						GAIndividual child;
						
						if( r.nextDouble() < recombProb )
							child = a.recombine( b );
						else 
							child = a;
						child = child.mutate();
						child.getCost();
						return child;
					}
				}));	
			}
			es.shutdown();
			
			gen.clear();
			for( Future<GAIndividual> f : futures ) {
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
		log.debug(k);
		return best;
	}
		
	// tournament selection
	public GAIndividual tournament( List<GAIndividual> gen, int k ) {
		List<GAIndividual> ng = new ArrayList<GAIndividual>();
		
		double sum = 0;
		for( int i = 0; i < k; i++ ) {
			GAIndividual in = gen.get( r.nextInt( gen.size() ) );
			ng.add( in );
			sum += in.getCost();
		}
		
		Collections.sort( ng );
		
		// deterministic
		return ng.get( 0 ); 		
	}
	
	public GAIndividual binaryProbabilisticTournament( List<GAIndividual> gen, double prob ) {
		Random r = new Random();
		GAIndividual a = gen.get( r.nextInt( gen.size() ) );
		GAIndividual b = gen.get( r.nextInt( gen.size() ) );
		
		if( b.getCost() < a.getCost() ) {
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
			sum += in.getCost();
				
		Random r = new Random();
		double v = r.nextDouble();
		
		double a = 0, b = 0;
		for( int j = 0; j < gen.size(); j++ ) {
			a = b;
			b = (sum - gen.get(j).getCost())/sum + b;
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
			sum += in.getCost();

		// intervals
		double ivs[] = new double[gen.size()+1];
		ivs[0] = 0.0f;
		for( int j = 0; j < ivs.length-1; j++ )  
			ivs[j+1] = sum - gen.get(j).getCost() + ivs[j];
		
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
		List<GAIndividual> init = new ArrayList<GAIndividual>();
		while( init.size() < 20 )
			init.add( new IncLLMGAIndividual() );
		GeneticAlgorithm gen = new GeneticAlgorithm();
		IncLLMGAIndividual result = (IncLLMGAIndividual)gen.search( init );
		
	}
}
