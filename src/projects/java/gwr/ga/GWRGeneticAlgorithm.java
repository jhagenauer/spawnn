package gwr.ga;

import java.io.File;
import java.util.AbstractMap.SimpleEntry;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import nnet.SupervisedUtils;
import spawnn.utils.DataUtils;
import spawnn.utils.GeoUtils.GWKernel;
import spawnn.utils.SpatialDataFrame;

public class GWRGeneticAlgorithm<T extends GAIndividual> {
	
	private static Logger log = Logger.getLogger(GWRGeneticAlgorithm.class);
	private final static Random r = new Random();
	int threads = Math.max(1 , Runtime.getRuntime().availableProcessors() -1 );;
	
	public int tournamentSize = 2;
	public double recombProb = 0.7;
				
	public T search( List<T> init, CostCalculator<T> cc ) {
		
		List<T> gen = new ArrayList<T>(init);
		Map<T,Double> costs = new HashMap<T,Double>(); // cost cache
		for( T i : init )
			costs.put(i, cc.getCost(i));
		
		T best = null;
		double bestCost = Double.MAX_VALUE;
				
		int noImpro = 0;
		int parentSize = init.size();
		int offspringSize = parentSize*2;
				
		int maxK = 500;
		int k = 0;
		while( /*k < maxK &&*/ noImpro < 100 ) {
			
			// check best and increase noImpro
			noImpro++;
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for( T cur : gen ) {
				if( best == null || costs.get(cur) < bestCost  ) { 
					best = cur;
					noImpro = 0;
				}
				ds.addValue( costs.get(cur) );
			}
			if( noImpro == 0 || k % 100 == 0 ) {
				log.info(k+","+ds.getMin()+","+ds.getMean()+","+ds.getMax()+","+ds.getStandardDeviation() );
			}
															
			// SELECT NEW GEN/POTENTIAL PARENTS
			// elite
			Collections.sort( gen, new Comparator<GAIndividual>() {
				@Override
				public int compare(GAIndividual g1, GAIndividual g2) {
					return Double.compare(costs.get(g1), costs.get(g2));
				}
			} );	
			
			List<T> elite = new ArrayList<T>();
			//elite.addAll( gen.subList(0, Math.max( 1, (int)( 0.01*gen.size() ) ) ) );
			gen.removeAll(elite);
				
			// SELECT PARENT
			List<T> selected = new ArrayList<T>(elite);
			while( selected.size() < parentSize ) {
				T i = tournament( gen, tournamentSize, costs );			
				selected.add( i );
			}		
			gen = selected;	
			
			// GENERATE OFFSPRING
			ExecutorService es = Executors.newFixedThreadPool(threads);
			List<Future<SimpleEntry<T, Double>>> futures = new ArrayList<Future<SimpleEntry<T, Double>>>();
			
			for( int i = 0; i < offspringSize; i++ ) {
				final T a = gen.get( r.nextInt( gen.size() ) );
				final T b = gen.get( r.nextInt( gen.size() ) );
				
				futures.add( es.submit( new Callable<SimpleEntry<T, Double>>() {
					@Override
					public SimpleEntry<T, Double> call() throws Exception {
						GAIndividual child;
						if( r.nextDouble() < recombProb )
							child = a.recombine( b );
						else 
							child = a;
						T mutChild = (T)child.mutate();
						
						return new SimpleEntry<T, Double>(mutChild, cc.getCost(mutChild));
					}
				}));	
			}
			es.shutdown();
			
			gen.clear();
			for( Future<SimpleEntry<T, Double>> f : futures ) {
				try {
					SimpleEntry<T, Double> e = f.get();
					gen.add( e.getKey() );
					costs.put( e.getKey(), e.getValue() );					
				} catch (InterruptedException e) {
					e.printStackTrace();
				} catch (ExecutionException e) {
					e.printStackTrace();
				}
			}
			costs.keySet().retainAll(gen);
			
			k++;
		}		
		log.debug(k);
		return best;
	}
		
	// tournament selection
	private T tournament( List<T> gen, int k, Map<T,Double> costs ) {
		List<T> ng = new ArrayList<T>();
		
		double sum = 0;
		for( int i = 0; i < k; i++ ) {
			T in = gen.get( r.nextInt( gen.size() ) );
			ng.add( in );
			sum += costs.get(in);
		}
		
		Collections.sort( ng, new Comparator<T>() {
			@Override
			public int compare(T g1, T g2) {
				return Double.compare(costs.get(g1), costs.get(g2));
			}
		} );	
		
		// deterministic
		return ng.get( 0 ); 		
	}
	
	private T binaryProbabilisticTournament( List<T> gen, double prob, Map<T,Double> costs ) {
		Random r = new Random();
		T a = gen.get( r.nextInt( gen.size() ) );
		T b = gen.get( r.nextInt( gen.size() ) );
		
		if( costs.get(b) < costs.get(a) ) {
			T tmp = a;
			a = b;
			b = tmp;
		}
		if( r.nextDouble() < prob )
			return a;
		else
			return b;
	}
	
	// roulette wheel selection
	private T rouletteWheelSelect( List<T> gen, Map<T,Double> costs ) {
		double sum = 0;
		for( T in : gen )
			sum += costs.get(in);
				
		Random r = new Random();
		double v = r.nextDouble();
		
		double a = 0, b = 0;
		for( int j = 0; j < gen.size(); j++ ) {
			a = b;
			b = (sum - costs.get(gen.get(j)))/sum + b;
			if( a <= v && v <= b || j+1 == gen.size() && a <= v ) 
				return gen.get(j);
		}
		return null;
	}

	// stochastic universal sampling
	private List<T> sus( List<T> gen, int n, Map<T,Double> costs ) {
		List<T> l = new ArrayList<T>();
		Collections.sort( gen, new Comparator<T>() {
			@Override
			public int compare(T g1, T g2) {
				return Double.compare(costs.get(g1), costs.get(g2));
			}
		} );	
		
		double sum = 0;
		for( T in : gen )
			sum += costs.get(in);

		// intervals
		double ivs[] = new double[gen.size()+1];
		ivs[0] = 0.0f;
		for( int j = 0; j < ivs.length-1; j++ )  
			ivs[j+1] = sum - costs.get(gen.get(j)) + ivs[j];
		
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
		
		GWKernel kernel = GWKernel.bisquare;
		int bwInit = 22;
		int[] ga = new int[]{0,1};
		int[] fa = new int[]{3};
		int ta = 4;
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromCSV(new File("output/spDat.csv"), new int[]{0,1}, new int[]{}, true);
		List<Entry<List<Integer>, List<Integer>>> cvList = SupervisedUtils.getCVList(10, 1, sdf.size() );
		 
		CostCalculator<GWRIndividual> cc_AICc = new GWRIndividualCostCalculator_AICc(sdf, fa, ga, ta, kernel);
		CostCalculator<GWRIndividual> cc_CV = new GWRIndividualCostCalculator_CV(sdf, cvList, fa, ga, ta, kernel);
				
		List<Integer> bw = new ArrayList<Integer>();
		for(int i = 0; i < sdf.samples.size(); i++ )
			bw.add( bwInit );
		GWRIndividual ind = new GWRIndividual( bw );
		log.debug("basic GWR AICc: "+cc_AICc.getCost(ind) );
		log.debug("basic GWR CV: "+cc_CV.getCost(ind) );

		List<GWRIndividual> init = new ArrayList<GWRIndividual>();
		while( init.size() < 25 ) {
			List<Integer> bandwidth = new ArrayList<>();
			while( bandwidth.size() < sdf.samples.size() )
				bandwidth.add( bwInit + r.nextInt( 8 ) - 4 );
			init.add( new GWRIndividual( bandwidth ) );
		}
		
		{
			log.debug("AICc");
			
			GWRGeneticAlgorithm<GWRIndividual> gen = new GWRGeneticAlgorithm<GWRIndividual>();
			GWRIndividual result = (GWRIndividual)gen.search( init, cc_AICc );
			log.debug("result GWR AICc: "+cc_AICc.getCost(result) );
			log.debug("result GWR CV: "+cc_CV.getCost(result) );
			
			List<double[]> r = new ArrayList<double[]>();
			for( int i = 0; i < sdf.samples.size(); i++ ) {
				double[] d = sdf.samples.get(i);
				r.add( new double[]{ d[0], d[1], result.getBandwidth().get(i) } );
			}
			DataUtils.writeCSV("output/result_AICc.csv", r, new String[]{"long","lat","b"});
		}
		
		{
			log.debug("CV");
			
			GWRGeneticAlgorithm<GWRIndividual> gen = new GWRGeneticAlgorithm<GWRIndividual>();
			GWRIndividual result = (GWRIndividual)gen.search( init, cc_CV );
			log.debug("result GWR AICc: "+cc_AICc.getCost(result) );
			log.debug("result GWR CV: "+cc_CV.getCost(result) );
			
			List<double[]> r = new ArrayList<double[]>();
			for( int i = 0; i < sdf.samples.size(); i++ ) {
				double[] d = sdf.samples.get(i);
				r.add( new double[]{ d[0], d[1], result.getBandwidth().get(i) } );
			}
			DataUtils.writeCSV("output/result_CV.csv", r, new String[]{"long","lat","b"});
		}
	}
}
