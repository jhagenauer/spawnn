
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
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
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.ColorBrewerUtil.ColorMode;
import spawnn.utils.DataFrame;
import spawnn.utils.DataUtils;

public class IrisTest {

	private static Logger log = Logger.getLogger(IrisTest.class);
	
	public static void main(String[] args) {
		Random r = new Random();
		int T_MAX = 100000;
		DataFrame df = DataUtils.readDataFrameFromCSV(new File("data/iris.csv"), new int[]{}, true);
		
		Map<double[],Integer> cMap = new HashMap<double[],Integer>();
		for( double[] d : df.samples )
			cMap.put( Arrays.copyOf(d, d.length-1), (int)d[d.length-1]);
		
		List<double[]> samples = new ArrayList<double[]>(cMap.keySet());			
		DataUtils.zScore(samples);
						
		Dist<double[]> eDist = new EuclideanDist();
		Grid2D<double[]> grid = new Grid2DHex<double[]>(12, 8);
		SomUtils.initRandom(grid, samples);
		
		log.debug("max radius: "+grid.getMaxDist());
			
		BmuGetter<double[]> bmuGetter = new DefaultBmuGetter<double[]>(eDist);
		
		SOM som = new SOM( new GaussKernel( new LinearDecay( 10, 1 ) ), new LinearDecay( 1,0.0 ), grid, bmuGetter );
		for (int t = 0; t < T_MAX; t++) {
			double[] x = samples.get(r.nextInt(samples.size()));
			som.train( (double)t/T_MAX, x );
		}
									
		log.debug("qe: "+SomUtils.getMeanQuantError( grid, bmuGetter, eDist, samples ) );
		log.debug("te: "+ SomUtils.getTopoError( grid, bmuGetter, samples ) );
				
		long time = System.currentTimeMillis();
		log.debug("spearman topo: "+SomUtils.getTopoCorrelation(samples, grid, bmuGetter, eDist, SomUtils.SPEARMAN_TYPE) );
		log.debug("Took: "+(System.currentTimeMillis()-time)+"ms");
		
		log.debug("pearson topo: "+SomUtils.getTopoCorrelation(samples, grid, bmuGetter, eDist, SomUtils.PEARSON_TYPE) );
		
		try {
			Map<GridPos,Set<double[]>> bmus = SomUtils.getBmuMapping(samples, grid, bmuGetter );
			SomUtils.printUMatrix( grid, eDist, ColorMode.Greys, SomUtils.HEX_UMAT, "output/irisUMatrix.png" );	
			SomUtils.printDMatrix(grid,eDist, ColorMode.Greys, new FileOutputStream("output/irisDMatrix.png"));
			
			SomUtils.printClassDist(cMap,bmus,grid,new FileOutputStream("output/class.png"));
			
			Collection<Set<GridPos>> clusters = SomUtils.getWatershedHex(140, 255, 0.5, grid, eDist, false);
			log.debug("clusters: "+clusters.size() );
			SomUtils.printClusters(clusters, grid, new FileOutputStream("output/watershed.png") );
			
			
			SomUtils.saveGrid(grid, new FileOutputStream("output/grid.xml"));
			grid = new Grid2DHex<double[]>( SomUtils.loadGrid(new FileInputStream("output/grid.xml")) );
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
}
