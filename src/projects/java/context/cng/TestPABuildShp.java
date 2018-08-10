package context.cng;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Geometry;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.NG;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

public class TestPABuildShp {

	/* Erzeuge output für map-erstellung
	 */
	
	private static Logger log = Logger.getLogger(TestPABuildShp.class);

	public static void main(String args[]) {
		
		Random r = new Random();

		final int T_MAX = 100000;			
		
		SpatialDataFrame sd = DataUtils.readShapedata(new File("data/marco/dat1/pgo_regression_transform.shp"), new int[]{}, false);
		final List<Geometry> geoms = sd.geoms;
		final List<double[]> samples = sd.samples;
		
		int[] fa = { 6, 7, 8, 9, 10, 11, 14 };
		int[] ga = { 4, 5 };

		Map<Integer, String> nmap = new HashMap<Integer, String>();
		nmap.put(4, "X");
		nmap.put(5, "Y");
		nmap.put(6, "LPI");
		nmap.put(7, "ATI");
		nmap.put(8, "CP05");
		nmap.put(9, "CP03");
		nmap.put(10, "MTV");
		nmap.put(11, "INM");
		nmap.put(14, "PTV");
				
		final Dist<double[]> geoDist = new EuclideanDist(ga );
		final Dist<double[]> fDist = new EuclideanDist(fa );
		
		DataUtils.normalizeColumns( samples, fa );
		DataUtils.normalizeGeoColumns( samples, ga ); 

		// geo som
		{
			int k = 1;
			Grid2D<double[]> grid = new Grid2DHex<double[]>(5, 1);
			SomUtils.initRandom(grid, samples);
			
			spawnn.som.bmu.BmuGetter<double[]> bmuGetter = new spawnn.som.bmu.KangasBmuGetter<double[]>(geoDist, fDist, k);
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
			
			/*try {
				Drawer.geoDrawCluster(cluster.values(), samples, geoms, new FileOutputStream("output/test_pa_"+grid.getPositions().size()+"_"+k+"_geosom.png"), true);
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}*/
			
			// add new class-column			
			List<double[]> ns = new ArrayList<double[]>();
			for( double[] d : samples ) {
				
				double[] nd = new double[fa.length+1];
				for( int i = 0; i < fa.length; i++ ) {
					nd[i] = d[fa[i]];
				}
														
				// find cluster
				int i = 0;
				for( Set<double[]> s : cluster.values() ) {
					if( s.contains(d)) {
						nd[nd.length-1] = i;
						break;
					}
					i++;
				}
				
				ns.add(nd);
			}
			
			String[] names = new String[fa.length+1];
			for( int i = 0; i < names.length-1; i++ )
				names[i] = nmap.get( fa[i] );
			names[names.length-1] = "PREDICTED";
						
			DataUtils.writeShape(ns, geoms, names, sd.crs, "output/test_pa_"+grid.getPositions().size()+"_"+k+"_geosom.shp" );
			
			// for boxplot out
			FileWriter fw;
			 try {
			  fw = new FileWriter( "output/test_pa_geosom_attr.csv" );
			  fw.write("attribute,value,cluster\n");
			  for( double[] d : ns ) {
				  for( int i = 0; i < fa.length; i++ ) {
					  fw.write(names[i]+","+d[i]+","+d[d.length-1]+"\n");
				  }
		      }
			  fw.close();
			} catch (IOException ex) {
				ex.printStackTrace();
			}
		}

		// cng
		{
			int k = 3;
			Sorter bmuGetter = new KangasSorter(geoDist, fDist, k);
			NG ng = new NG(5, 5/2, 0.01, 0.5, 0.005, samples.get(0).length, bmuGetter);

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
			
			/*try {
				Drawer.geoDrawCluster(cluster.values(), samples, geoms, new FileOutputStream("output/test_pa_"+ng.getNeurons().size()+"_"+k+"_cng.png"), true);
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}*/
			
			// add new class-column			
			List<double[]> ns = new ArrayList<double[]>();
			for( double[] d : samples ) {
				
				double[] nd = new double[fa.length+1];
				for( int i = 0; i < fa.length; i++ ) {
					nd[i] = d[fa[i]];
				}
														
				// find cluster
				int i = 0;
				for( Set<double[]> s : cluster.values() ) {
					if( s.contains(d)) {
						nd[nd.length-1] = i;
						break;
					}
					i++;
				}
				
				ns.add(nd);
			}
			
			String[] names = new String[fa.length+1];
			for( int i = 0; i < names.length-1; i++ )
				names[i] = nmap.get( fa[i] );
			names[names.length-1] = "PREDICTED";

			DataUtils.writeShape(ns, geoms, names, sd.crs, "output/test_pa_"+ng.getNeurons().size()+"_"+k+"_cng.shp");
			
			// for boxplot out
			FileWriter fw;
			 try {
			  fw = new FileWriter( "output/test_pa_cng_attr.csv" );
			  fw.write("attribute,value,cluster\n");
			  for( double[] d : ns ) {
				  for( int i = 0; i < fa.length; i++ ) {
					  fw.write(names[i]+","+d[i]+","+d[d.length-1]+"\n");
				  }
		      }
			  fw.close();
			} catch (IOException ex) {
				ex.printStackTrace();
			}
		}
	}
}
