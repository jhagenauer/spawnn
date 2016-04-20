package regionalization.ga;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
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

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.Clustering;
import spawnn.utils.Clustering.HierarchicalClusteringType;
import spawnn.utils.DataUtils;
import spawnn.utils.RegionUtils;
import spawnn.utils.SpatialDataFrame;

public class GeneticAlgorithm {
	
	private static Logger log = Logger.getLogger(GeneticAlgorithm.class);
	private final static Random r = new Random();
	int threads = 4;
	
	public int tournamentSize = 3;
	public double recombProb = 0.6;
	
	public static int mode = 4;
			
	public GAIndividual search( List<GAIndividual> init ) {		
		List<GAIndividual> gen = new ArrayList<GAIndividual>(init);
		GAIndividual best = null;
		
		int noImpro = 0;
		int parentSize = init.size();
		int offspringSize = parentSize*2;
				
		int maxK = 1000;
		int k = 0;
		while( k < maxK /*|| noImpro < 100*/ ) {
				
			if( mode == 0 )
				RegioGAIndividual.probSeedGenMod = 1 - (double)k/1000;
			else if( mode == 1 )
				RegioGAIndividual.probSeedGenMod = Math.pow( 0.001, (double)k/1000 ) ;
			else if( mode == 2 )
				RegioGAIndividual.probSeedGenMod = 1.0 - Math.pow( (double)k/1000, 2.0 ) ;
			else if( mode == 3 )
				RegioGAIndividual.probSeedGenMod = -1;
			else 
				RegioGAIndividual.probSeedGenMod = 0; // no seed ever
			
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
				;//log.info(k+","+ds.getMin()+","+ds.getMean()+","+ds.getMax()+","+ds.getStandardDeviation() );
															
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
		int numRegions = 4;
		
		/*int size = 200;
		SpatialDataFrame sfd = DataUtils.readSpatialDataFrameFromShapefile(new File("data/redcap/Election/election2004.shp"), true );
		final double[] mean = new double[2];
		for (int i = 0; i < sfd.samples.size(); i++) {
			double[] d = sfd.samples.get(i);
			Point p1 = sfd.geoms.get(i).getCentroid();
			
			// ugly, but easy
			d[0] = p1.getX();
			d[1] = p1.getY();
			mean[0] += d[0]/sfd.samples.size();
			mean[1] += d[1]/sfd.samples.size();
		}
		
		int[] fa = new int[] { 7 };
		final Dist<double[]> gDist = new EuclideanDist(new int[]{ 0, 1});
		Dist<double[]> fDist = new EuclideanDist( fa );

		List<double[]> samples = new ArrayList<double[]>(sfd.samples);
		Collections.sort(samples, new Comparator<double[]>() {
			@Override
			public int compare(double[] o1, double[] o2) {
				return Double.compare( gDist.dist(o1, mean), gDist.dist(o2,mean) );
			}
		});
		
		samples = samples.subList(0, size);
		List<Geometry> geoms = new ArrayList<Geometry>();
		for( double[] d : samples )
			geoms.add( sfd.geoms.get(sfd.samples.indexOf(d)));*/
		
		int size = 200;
		String fn = "output/"+size+"rnd.shp";
		//SampleBuilder.buildArtifialSpatialDataSet(size, fn );
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File( fn ), true );
		List<double[]> samples = sdf.samples;
		List<Geometry> geoms = sdf.geoms;
		
		int[] fa = new int[]{ 2 };
		Dist<double[]> fDist = new EuclideanDist( fa );
				
		final Map<double[], Set<double[]>> cm = RegionUtils.deriveQueenContiguitiyMap(samples, geoms);	
		log.debug("#regions: "+RegionUtils.getAllContiguousSubcluster(cm, new HashSet<double[]>(samples)).size());
		
		for (int i : fa)
			DataUtils.zScoreColumn(samples, i);
		
		DescriptiveStatistics ds = new DescriptiveStatistics();
		
		for( int i = 0; i < 25; i++ ) {
			log.debug("i: "+i);
			
			List<GAIndividual> init = new ArrayList<GAIndividual>();
			while( init.size() < 25 ) {
				List<double[]> chromosome = new ArrayList<double[]>(samples);
				chromosome.add(1, chromosome.remove(5));
				chromosome.add(2, chromosome.remove(8));
				chromosome.add(3, chromosome.remove(126));
				//Collections.shuffle(chromosome);
				init.add( new RegioGAIndividual( chromosome, numRegions, new WCSSCostCalulator(fDist), cm ) );
			}
			
			GeneticAlgorithm gen = new GeneticAlgorithm();
			RegioGAIndividual result = (RegioGAIndividual)gen.search( init );
			
			/*try {
				Drawer.geoDrawCluster(result.getCluster(), samples, geoms, new FileOutputStream("output/result"+i+".png"), true);
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}*/
			
			ds.addValue(DataUtils.getWithinClusterSumOfSuqares( result.getCluster(), fDist));
		}
									
		log.debug("GA WCSS: "+ ds.getMin()+","+ds.getMean()+","+ds.getMax()+","+ds.getStandardDeviation() );
			
		for( HierarchicalClusteringType type : HierarchicalClusteringType.values() ) {
			List<Set<double[]>> wardCluster = Clustering.cutTree(Clustering.getHierarchicalClusterTree(cm, fDist, HierarchicalClusteringType.ward), numRegions);
			String str = "";
			for( Set<double[]> s : wardCluster ) {
				int idx = Integer.MAX_VALUE;
				for( double[] d : s )
					idx = Math.min(idx, samples.indexOf(d) );
				str += idx +" ";
			}
			log.debug(type+", WCSS: "+DataUtils.getWithinClusterSumOfSuqares(wardCluster, fDist) +": "+str );
		}
	}
}
