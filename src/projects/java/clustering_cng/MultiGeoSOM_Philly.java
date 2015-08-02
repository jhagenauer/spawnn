package clustering_cng;

import java.io.File;
import java.util.ArrayList;
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
import spawnn.utils.SpatialDataFrame;

import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.Point;

public class MultiGeoSOM_Philly {

	private static Logger log = Logger.getLogger(MultiGeoSOM_Philly.class);
	
	public static void main(String[] args) {
		
		File file = new File("data/philadelphia/tracts/philadelphia_tracts_with_pop.shp");
		final SpatialDataFrame sd = DataUtils.readShapedata(file, new int[] {}, false);

		List<double[]> samplesOrig = sd.samples;
		List<Geometry> geoms = sd.geoms;
		
		// ugly, but easy
		for (int i = 0; i < samplesOrig.size(); i++) {
			double[] d = samplesOrig.get(i);
			Point p = geoms.get(i).getCentroid();
			d[0] = p.getX();
			d[1] = p.getY();
		}

		final List<double[]> samples = new ArrayList<double[]>();
		for (double[] d : samplesOrig) {

			double[] nd = new double[] { 
					samples.size(), // id
					d[0], // x
					d[1], // y

					d[2], // pop

					// ---- age ----
					(d[3] + d[4] + d[5] + d[6] + d[7]) / d[2], // age 0 to 24
					(d[8] + d[9] + d[10] + d[11] + d[12] + d[13] + d[14] + d[15]) / d[2], // age 25 to 64
					(d[16] + d[17] + d[18] + d[19] + d[20]) / d[2], // age 65 and older
					// d[59], // median age

					// ---- race ----
					d[79] / d[2], // white
					d[80] / d[2], // black
					// d[81], // indian
					d[82] / d[2], // asian
					d[115] / d[2], // hispanic
					// d[124], // white, not hispanic
					// d[125], // black, not hispanic

					// ---- households ----
					// d[151], // households
					// d[152], // family households
					d[168], // avg household size
					// d[169], // avg family size

					// ---- housing ----
					// d[170], // total housing units
					// d[171], // occupied
					// d[172], // vacant
					d[171] / d[170], // occupied-rate

					// d[181], // total occupied housing units
					// d[182], // owner-occupied housing units
					d[183] / d[170], // renter-occupied housing units
			};

			// "repair" NANs
			for (int i = 0; i < nd.length; i++)
				if (Double.isNaN(nd[i]))
					nd[i] = 0;

			samples.add(nd);
		}
		
		final String header[] = new String[] { "id2", "x", "y", "pop", "0to24", "25to64", "65older", "white", "black", "asian", "hispanic", "avgHHSize", "occup", "rentOccup" };

		int[] fa = new int[samples.get(0).length - 3];
		for (int i = 3; i < fa.length; i++)
			fa[i] = i;

		int[] ga = { 1, 2 };
		
		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa);
		
		DataUtils.zScoreColumns(samples, fa);

		final Random r = new Random();
		final int T_MAX = 100000;

	
		for( int k = 0;; k++ ) {
			log.debug("k: "+k);
			BmuGetter<double[]> bg = new KangasBmuGetter<double[]>(gDist, fDist, k );
			Grid2DHex<double[]> grid = new Grid2DHex<double[]>(15, 10);
			SomUtils.initRandom(grid, samples);
			
			int maxDist = grid.getMaxDist();
			if( k > maxDist )
				break;
			
			SOM som = new SOM( new GaussKernel( new LinearDecay( maxDist, 1 ) ), new LinearDecay( 1,0.0 ), grid, bg );	
			for (int t = 0; t < T_MAX; t++) {
				double[] x = samples.get(r.nextInt(samples.size() ) );
				som.train( (double)t/T_MAX, x);
			}
			
			SomUtils.printUMatrix(grid, fDist, "output/geosom_"+k+".png");
		}
	}
}
