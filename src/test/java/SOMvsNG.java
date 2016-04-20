

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.NG;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;

public class SOMvsNG {
	
	private static Logger log = Logger.getLogger(SOMvsNG.class);
	
	public static void main( String args[] ) {
		final Random r = new Random();
		
		final int T_MAX = 100000;
		
		//List<double[]> samples = SampleBuilder.readSamplesFromFcps("data/fcps/GolfBall.lrn", "data/fcps/GolfBall.cls");
		//List<double[]> samples = SampleBuilder.readSamplesFromFcps("data/fcps/Chainlink.lrn", "data/fcps/Chainlink.cls");
		List<double[]> samples = null;
		try {
			samples = DataUtils.readCSV( new FileInputStream("data/wine.csv") );
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		samples = new ArrayList<double[]>();
		for( int i = 0; i < 1780; i++ ) {
			double[] d = new double[14];
			for( int j = 0; j < d.length; j++ )
				d[j] = r.nextDouble();
			samples.add( d );
		}
		
		// int[] fa = new int[]{0,1,2};
		int[] fa = new int[]{0,1,2,3,4,5,6,7,8,9,10,11,12,13};
			
		DataUtils.zScoreColumns(samples, fa);
		//DataUtil.normalizeColumns(samples, fa);
						
		final Dist eDist = new EuclideanDist();
		Dist fDist = new EuclideanDist(fa);
				
		// SOM
		{					
		Grid2D<double[]> grid = new Grid2DHex<double[]>(5, 5);
												
		//grid.initLinear(samples, true);
		spawnn.som.bmu.BmuGetter<double[]> bmuGetter = new spawnn.som.bmu.DefaultBmuGetter( fDist );
		SOM som = new SOM( new GaussKernel( grid.getMaxDist() ), new LinearDecay(0.5,0.0), grid, bmuGetter );
						
		for (int t = 0; t < T_MAX; t++) {
			double[] x = samples.get(r.nextInt(samples.size() ) );
			som.train( (double)t/T_MAX, x );
		}
					
		Map<double[],Set<double[]>> cluster = new HashMap<double[],Set<double[]>>();
		for( double[] d : samples ) {
			double[] bmu = grid.getPrototypeAt( bmuGetter.getBmuPos(d, grid ) );
			if( !cluster.containsKey(bmu) )
				cluster.put(bmu, new HashSet<double[]>() );
			cluster.get(bmu).add(d);
		}
				
		log.debug("som:");
		log.debug("qe: "+DataUtils.getMeanQuantizationError( cluster, fDist ) );
		log.debug("te: "+SomUtils.getTopoError(grid, bmuGetter, samples) );
		log.debug("sse: "+DataUtils.getSumOfSquares(cluster, fDist));
		}
		
		// cng
		{	
		Sorter bmuGetter = new DefaultSorter( fDist );
		NG ng = new NG( 25, 10, 0.01, 0.5, 0.005, samples.get(0).length, bmuGetter );
		ng.initRandom(samples);			
		
		for (int t = 0; t < T_MAX; t++) {
			double[] x = samples.get( r.nextInt(samples.size() ) );
			ng.train( (double)t/T_MAX, x );
		}
						
		Map<double[],Set<double[]>> cluster = new HashMap<double[],Set<double[]>>();
		for( double[] w : ng.getNeurons() )
			cluster.put( w, new HashSet<double[]>() );
		for( double[] d : samples ) {
			bmuGetter.sort(d, ng.getNeurons() );
			double[] bmu = ng.getNeurons().get(0);
			cluster.get(bmu).add(d);
		}
							
		log.debug("cng:");
		log.debug("qe: "+DataUtils.getMeanQuantizationError( cluster, fDist ) );	
		log.debug("ss: "+DataUtils.getSumOfSquares(cluster, fDist));
		}
	}
}
