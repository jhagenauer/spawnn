package context.cng;

import java.io.File;
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
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2D_Map;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.SOM;
import spawnn.utils.ClusterValidation;
import spawnn.utils.DataUtils;

public class TestClusterBuildShp {

	/* Erzeuge output für map-erstellung
	 */
	
	private static Logger log = Logger.getLogger(TestClusterBuildShp.class);

	public static void main(String args[]) {

		List<double[]> samples = DataUtils.readSamplesFromShapeFile(new File("data/cng/test2a_nonoise.shp"), new int[] {}, true);
		//List<Geometry> geoms = DataUtil.readGeometriesFromShapeFile( new File("data/cng/test2a.shp") );

		final Map<Integer, Set<double[]>> classes = new HashMap<Integer, Set<double[]>>();
		for (double[] d : samples) {
			int c = (int) d[3];
			if (!classes.containsKey(c))
				classes.put(c, new HashSet<double[]>());
			classes.get(c).add(d);
		}
		
		final int numCluster = classes.size();

		final int[] fa = { 2 };
		final int[] ga = new int[] { 0, 1 };

		final Dist eDist = new EuclideanDist();
		final Dist geoDist = new EuclideanDist(ga);
		final Dist fDist = new EuclideanDist(fa);

		DataUtils.normalizeColumns(samples, fa);

		final Random r = new Random();
		final int T_MAX = 100000;

		// geo som
		{

			Grid2D_Map<double[]> grid = new Grid2DHex<double[]>(5, 5);
			// grid.initLinear(samples, true);
			spawnn.som.bmu.BmuGetter<double[]> bmuGetter = new spawnn.som.bmu.KangasBmuGetter(geoDist, fDist, 1);
			SOM som = new SOM(new GaussKernel(grid.getMaxDist()), new LinearDecay(0.5, 0.0), grid, bmuGetter);

			for (int t = 0; t < T_MAX; t++) {
				double[] x = samples.get(r.nextInt(samples.size()));
				som.train((double) t / T_MAX, x);
			}

			Map<double[], Set<double[]>> cluster = new HashMap<double[], Set<double[]>>();
			for (double[] d : samples) {
				double[] bmu = grid.getPrototypeAt(bmuGetter.getBmuPos(d, grid));
				if (!cluster.containsKey(bmu))
					cluster.put(bmu, new HashSet<double[]>());
				cluster.get(bmu).add(d);
			}

			log.debug("geosom: ");
			log.debug("qe: "+ DataUtils.getMeanQuantizationError(cluster, fDist) );
			log.debug("ge: "+ DataUtils.getMeanQuantizationError(cluster, geoDist) );
			log.debug("nmi: "+ ClusterValidation.getNormalizedMutualInformation(cluster.values(), classes.values() ) );
			
			/*try {
				Drawer.geoDrawCluster(cluster.values(), samples, geoms, new FileOutputStream("output/cluster_geosom.png"), true);
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}
			
			// add new class-column			
			List<double[]> ns = new ArrayList<double[]>();
			for( double[] d : samples ) {
				double[] nd = Arrays.copyOf(d, d.length+1);
				
				// find cluster
				int i = 0;
				for( Set<double[]> s : cluster.values() ) {
					if( s.contains(d))
						nd[nd.length-1] = i;
					i++;
				}
				ns.add(nd);
			}
			
			DataUtil.writeToShape(ns, geoms, new String[]{"X","Y","VALUE","CLUSTER","PREDICTED"}, "output/cluster_geosom.shp");*/
		}

		// cng
		{

			Sorter bmuGetter = new KangasSorter(geoDist, fDist, 3);
			NG ng = new NG(numCluster, numCluster/2, 0.01, 0.5, 0.005, samples.get(0).length, bmuGetter);

			for (int t = 0; t < T_MAX; t++) {
				double[] x = samples.get(r.nextInt(samples.size()));
				ng.train((double) t / T_MAX, x);
			}

			Map<double[], Set<double[]>> cluster = new HashMap<double[], Set<double[]>>();
			for (double[] w : ng.getNeurons())
				cluster.put(w, new HashSet<double[]>());
			for (double[] d : samples) {
				bmuGetter.sort(d, ng.getNeurons());
				double[] bmu = ng.getNeurons().get(0);
				cluster.get(bmu).add(d);
			}

			log.debug("cng: ");
			log.debug("qe: "+ DataUtils.getMeanQuantizationError(cluster, fDist) );
			log.debug("ge: "+ DataUtils.getMeanQuantizationError(cluster, geoDist) );
			log.debug("nmi: "+ ClusterValidation.getNormalizedMutualInformation(cluster.values(), classes.values() ) );
			
			/*try {
				Drawer.geoDrawCluster(cluster.values(), samples, geoms, new FileOutputStream("output/cluster_cng.png"), true);
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}
			
			// add new class-column			
			List<double[]> ns = new ArrayList<double[]>();
			for( double[] d : samples ) {
				double[] nd = Arrays.copyOf(d, d.length+1);
				
				// find cluster
				int i = 0;
				for( Set<double[]> s : cluster.values() ) {
					if( s.contains(d))
						nd[nd.length-1] = i;
					i++;
				}
				ns.add(nd);
			}
			
			DataUtil.writeToShape(ns, geoms, new String[]{"X","Y","VALUE","CLUSTER","PREDICTED"}, "output/cluster_cng.shp");*/

		}
	}
}
