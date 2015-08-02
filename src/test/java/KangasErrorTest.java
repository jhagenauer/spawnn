

import java.util.List;
import java.util.Random;

import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.bmu.KangasBmuGetter;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;

public class KangasErrorTest {
	
	private static Logger log = Logger.getLogger(KangasErrorTest.class);
	
	public static void main(String args[]) {
		final Random r = new Random();
		
		int xDim = 15, yDim = 10;

		final int T_MAX = 100000;
		List<double[]> samples = DataUtils.readCSV("data/squareville.csv");
		/*List<double[]> samples = new ArrayList<double[]>();
		for( int i = 0; i < 10000; i++ ) {
			double x = r.nextDouble();
			double y = r.nextDouble();
			samples.add( new double[]{ x,y,x+y } );
		}*/
		
		final int[] ga = new int[] { 0, 1 };
		final int[] fa = { 2 };

		final Dist<double[]> eDist = new EuclideanDist();
		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa);
		
		int max_r = Integer.MAX_VALUE;
		for( int i = 0; i < max_r; i++ ) {
			log.debug("i: "+i);
			Grid2D<double[]> grid = new Grid2DHex<double[]>(xDim,yDim);
			SomUtils.initRandom(grid, samples);
			max_r = grid.getMaxDist();
			
			KangasBmuGetter<double[]> bg = new KangasBmuGetter<double[]>(gDist, eDist, i);
			
			SOM som = new SOM( new GaussKernel(grid.getMaxDist()), new LinearDecay(0.5,0.0), grid, bg );
			for (int t = 0; t < T_MAX; t++) {
				double[] x = samples.get(r.nextInt(samples.size() ) );
				som.train( (double)t/T_MAX, x );
			}
			
			log.debug("fqe: " + SomUtils.getMeanQuantError(grid, bg, fDist, samples));
			log.debug("gqe: " + SomUtils.getMeanQuantError(grid, bg, gDist, samples));
			//log.debug("te: " + SomUtils.getTopoError(grid, bg, samples));
			log.debug("ke: "+SomUtils.getKangasError(samples, grid, bg) );
			//log.debug("spearman ftopo: "+SomUtils.getTopoCorrelation(samples, grid, bg, fDist, SomUtils.SPEARMAN_TYPE));
			//log.debug("spearm gtopo: "+SomUtils.getTopoCorrelation(samples, grid, bg, gDist, SomUtils.SPEARMAN_TYPE));
			
			SomUtils.printUMatrix(grid, fDist, "output/umat"+i+".png");
		}
	}
}	
