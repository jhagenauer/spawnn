package clustering_cng;


import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import javax.xml.stream.XMLOutputFactory;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.XMLStreamWriter;

import org.apache.log4j.Logger;


import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.KangasBmuGetter;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.Point;

public class MultiGeoSOM_Diamond {

	private static Logger log = Logger.getLogger(MultiGeoSOM_Diamond.class);
	
	public static void main(String[] args) {
		
		SpatialDataFrame sd = DataUtils.readShapedata( new File("data/diamond/diamond.shp") , new int[] {}, true);
		final List<double[]> samples = sd.samples;
		final String[] header = sd.names.toArray(new String[]{} );
		
		final Map<double[],Integer> classes = new HashMap<double[],Integer>();
		for( double[] d : samples ) {
			classes.put(d, (int)d[3]);
		}
		
		for( int i = 0; i < header.length; i++ )
			header[i] = header[i].toLowerCase();
		
		final int[] ga = new int[] { 0, 1 };
		final int[] fa = { 2 };
		
		final Dist<double[]> gDist = new EuclideanDist( ga);
		final Dist<double[]> fDist = new EuclideanDist( fa);

		final Random r = new Random();
		final int T_MAX = 100000;

	
		for( int k = 0;; k++ ) {
			log.debug("k: "+k);
			BmuGetter<double[]> bg = new KangasBmuGetter<double[]>(gDist, fDist, k );
			Grid2DHex<double[]> grid = new Grid2DHex<double[]>(5, 5);
			SomUtils.initRandom(grid, samples);
			
			int maxDist = grid.getMaxDist();
			if( k > maxDist )
				break;
			
			SOM som = new SOM( new GaussKernel( new LinearDecay( maxDist, 1 ) ), new LinearDecay( 1,0.0 ), grid, bg );	
			for (int t = 0; t < T_MAX; t++) {
				double[] x = samples.get(r.nextInt(samples.size() ) );
				som.train( (double)t/T_MAX, x);
			}
			
			
			try {
				SomUtils.printUMatrix(grid, fDist, "output/umatrix_"+k+".png");
				SomUtils.printDMatrix(grid, fDist, new FileOutputStream("output/dmatrix_"+k+".png"));
				SomUtils.printClassDist(classes, SomUtils.getBmuMapping(samples, grid, bg), grid, new FileOutputStream("output/classdist_"+k+".png") );
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}
		}
	}
}
