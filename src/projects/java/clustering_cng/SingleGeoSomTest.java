package clustering_cng;


import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.List;
import java.util.Random;

import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.KangasBmuGetter;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;

public class SingleGeoSomTest {

	private static Logger log = Logger.getLogger(SingleGeoSomTest.class);
	
	public static void main(String[] args) {

		Random r = new Random();
		int T_MAX = 100000;

		List<double[]> samples = DataUtils.readSamplesFromShapeFile( new File("output/diamond.shp") , new int[] {}, true);
		final int[] ga = new int[] { 0, 1 };
		final int[] fa = { 2 };
		
		final Dist<double[]> eDist = new EuclideanDist();
		final Dist<double[]> gDist = new EuclideanDist( ga);
		final Dist<double[]> fDist = new EuclideanDist( fa);
		
		Grid2DHex<double[]> grid = new Grid2DHex<double[]>(12, 8 );
		SomUtils.initLinear(grid, samples, true);
			
		BmuGetter<double[]> bmuGetter = new KangasBmuGetter<double[]>(gDist, fDist, 2);
				
		SOM som = new SOM( new GaussKernel(grid.getMaxDist()), new LinearDecay(1.0,0.0), grid, bmuGetter );
		for (int t = 0; t < T_MAX; t++) {
			double[] x = samples.get(r.nextInt(samples.size() ) );
			//double[] x = samples.get(t%samples.size()); 
			som.train( (double)t/T_MAX, x );
		}
		
							
		log.debug("qe: "+SomUtils.getMeanQuantError( grid, bmuGetter, eDist, samples ) );
		log.debug("te: "+ SomUtils.getTopoError( grid, bmuGetter, samples ) );
		
		{
			long time = System.currentTimeMillis();
			log.debug("spearman topo: "+SomUtils.getTopoCorrelation(samples, grid, bmuGetter, eDist, SomUtils.SPEARMAN_TYPE));
			log.debug("Took: "+(System.currentTimeMillis()-time)+"ms");
		}
								
		try {
			SomUtils.printUMatrix( grid, eDist, new FileOutputStream( "output/umatrix.png" ) );
			SomUtils.printGeoGrid(new int[]{0,1}, grid, new FileOutputStream("output/topo.png") );
									
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
}
