package regionalization.ga;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Geometry;

import spawnn.dist.EuclideanDist;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.RegionUtils;

public class GeneticAlgorithm2 {
	
	private static Logger log = Logger.getLogger(GeneticAlgorithm2.class);
	private final static Random r = new Random();
	int threads = 20;
	
	public int trSize = 2;
	public double recombProb = 0.6;
				
	public GAIndividual search( List<GAIndividual> init ) {		
		List<GAIndividual> gen = new ArrayList<GAIndividual>(init);
		GAIndividual best = null;
		
		int noImpro = 0;
		int parentSize = init.size();
		int offspringSize = parentSize*2;
						
		int k = 0;
		while( k < 20000 ) {
			
			if( k % 500 == 0 )  {
				double avg = 0;
				for( GAIndividual i : gen ) 
					avg += (double)((RegioGAIndividual2)i).getRegionList().size()/gen.size(); 
				log.debug(avg);
			}
						
			// check best and increase noImpro
			noImpro++;
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for( GAIndividual cur : gen ) {
				if( best == null || cur.getValue() < best.getValue()  ) { 
					best = cur;
					noImpro = 0;
				}
				ds.addValue( cur.getValue() );
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
				GAIndividual i = tournament( gen, trSize );			
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
																	
						return child.mutate();
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
		return best;
	}
		
	// tournament selection
	public GAIndividual tournament( List<GAIndividual> gen, int k ) {
		List<GAIndividual> ng = new ArrayList<GAIndividual>();
		
		double sum = 0;
		for( int i = 0; i < k; i++ ) {
			GAIndividual in = gen.get( r.nextInt( gen.size() ) );
			ng.add( in );
			sum += in.getValue();
		}
		
		Collections.sort( ng );
		
		// deterministic
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
		//List<Geometry> geoms = DataUtil.readGeometriesFromShapeFile(new File("data/regionalization/200rand.shp"));
		//List<double[]> samples = DataUtil.readSamplesFromShapeFile(new File("data/regionalization/200rand.shp"), new int[] {}, true);
		List<Geometry> geoms = DataUtils.readGeometriesFromShapeFile(new File("data/regionalization/500rand.shp"));
		List<double[]> samples = DataUtils.readSamplesFromShapeFile(new File("data/regionalization/500rand.shp"), new int[] {}, true);
		//List<Geometry> geoms = DataUtil.readGeometriesFromShapeFile(new File("data/regionalization/1000rand.shp"));
		//List<double[]> samples = DataUtil.readSamplesFromShapeFile(new File("data/regionalization/1000rand.shp"), new int[] {}, true);
		//final List<Geometry> geoms = DataUtil.readGeometriesFromShapeFile(new File("data/redcap/Election/election2004.shp"));
		//final List<double[]> samples = DataUtil.readSamplesFromShapeFile(new File("data/redcap/Election/election2004.shp"), new int[] {}, true);
		
		int[] fa = new int[] { 7 };

		for (int i : fa)
			DataUtils.zScoreColumn(samples, i);
		
		//final Map<double[], Set<double[]>> cm = RegionUtils.readContiguitiyMap(samples, "data/regionalization/200rand.ctg");
		final Map<double[], Set<double[]>> cm = RegionUtils.readContiguitiyMap(samples, "data/regionalization/500rand.ctg");
		//final Map<double[], Set<double[]>> cm = RegionUtils.readContiguitiyMap(samples, "data/regionalization/1000rand.ctg");
		//final Map<double[], Set<double[]>> cm = RegionUtils.readContiguitiyMap(samples, "data/redcap/Election/election2004_Queen.ctg");
						
		List<GAIndividual> init = new ArrayList<GAIndividual>();
		while( init.size() < 50 ) {
			List<double[]> chromosome = new ArrayList<double[]>(samples);
			Collections.shuffle(chromosome);
			init.add( new RegioGAIndividual2( chromosome, new WCSSCostCalulator(new EuclideanDist(fa)), cm ) );
		}
		
		GeneticAlgorithm2 ga = new GeneticAlgorithm2();
		RegioGAIndividual2 result = (RegioGAIndividual2)ga.search( init );
		
									
		log.debug("Heterogenity: "+result.getValue() );
		log.debug("Regions: "+result.getRegionList().size());
						
		try { 
			Drawer.geoDrawCluster( result.getRegionList(), samples, geoms, new FileOutputStream("output/ga.png"), true ); 
		} catch(FileNotFoundException e) {
			e.printStackTrace(); 
		}	
	}
}
