


import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.DefaultBmuGetter;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;

public class SOMTest {

	private static Logger log = Logger.getLogger(SOMTest.class);
	
	public static void main(String[] args) {

		Random r = new Random();
		int T_MAX = 1000000;

		List<double[]> samples = DataUtils.readCSV("data/crime/dts.csv");
		
				
		Dist<double[]> eDist = new EuclideanDist();
		
		int vLength = samples.get(0).length;
		Grid2DHex<double[]> grid = new Grid2DHex<double[]>(45, 35 );
		//SomUtils.initLinear(grid, samples, true);
		SomUtils.initRandom(grid, samples);
			
		BmuGetter<double[]> bmuGetter = new DefaultBmuGetter<double[]>(eDist);
				
		SOM som = new SOM( new GaussKernel( new LinearDecay(grid.getMaxDist(), 1)), new LinearDecay(1.0,0.0), grid, bmuGetter );
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
			Map<GridPos,Set<double[]>> bmus = SomUtils.getBmuMapping(samples, grid, bmuGetter );
			SomUtils.printUMatrix( grid, eDist, new FileOutputStream( "output/umatrix.png" ) );
			SomUtils.printGeoGrid(new int[]{0,1}, grid, new FileOutputStream("output/topo.png") );
						
			/*for( int i = 0; i < vLength; i++ )
				SomUtils.printComponentPlane(grid, i, new FileOutputStream("output/component"+i+".png"));*/
						
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
}
