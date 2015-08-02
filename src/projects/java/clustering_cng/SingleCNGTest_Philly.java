package clustering_cng;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.io.Writer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import javax.xml.stream.XMLOutputFactory;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.XMLStreamWriter;

import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.Connection;
import spawnn.ng.NG;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.utils.DataUtils;

import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.Point;

public class SingleCNGTest_Philly {

	private static Logger log = Logger.getLogger(SingleCNGTest_Philly.class);

	public static void main(String args[]) {
		int N = 25;

		final Random r = new Random();

		final int T_MAX = 100000;

		/*
		 * File file = new File("data/census/Tract_2010Census_DP1.shp");
		 * List<double[]> samples = DataUtils.readSamplesFromShapeFile( file ,
		 * new int[] {}, true); List<Geometry> geoms =
		 * DataUtils.readGeometriesFromShapeFile(file); final int[] ga = new
		 * int[] { 0, 1 }; final int[] fa = { 2 };
		 */

		File file = new File("data/philadelphia/tracts/philadelphia_tracts_with_pop.shp");
		List<double[]> samples = DataUtils.readSamplesFromShapeFile(file, new int[] {}, false);
		List<Geometry> geoms = DataUtils.readGeometriesFromShapeFile(file);

		// ugly, but easy
		for (int i = 0; i < samples.size(); i++) {
			double[] d = samples.get(i);
			Point p = geoms.get(i).getCentroid();
			d[0] = p.getX();
			d[1] = p.getY();
		}

		// all
		/*
		 * List<Integer> fas = new ArrayList<Integer>(); for( int idx = 2; idx
		 * <= 187; idx++ ) fas.add(idx); Set<Integer> rm = new
		 * HashSet<Integer>(); for( int i : fas ) { for( int j : fas ) {
		 * 
		 * if( i == j ) continue;
		 * 
		 * boolean ident = true; for( double[] d : samples ) if( d[i] != d[j] )
		 * { ident = false; break; }
		 * 
		 * if( ident ) rm.add( Math.max(i,j) ); } }
		 * 
		 * log.debug("Double columns: "+rm+", removing them.");
		 * fas.removeAll(rm); log.debug("Remaining columns: "+fas.size());
		 * 
		 * int[] fa = new int[fas.size()]; for( int i = 0; i < fa.length; i++ )
		 * fa[i] = fas.get(i);
		 */

		List<double[]> ns = new ArrayList<double[]>();
		for (double[] d : samples) {

			double[] nd = new double[] { 
					d[0], // x
					d[1], // y

					d[2], // pop

					// ---- age ----
					(d[3] + d[4] + d[5] + d[6] + d[7]) / d[2], // age 0 to 24
					(d[8] + d[9] + d[10] + d[11] + d[12] + d[13] + d[14] + d[15]) / d[2], // age 24 to 64
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

			ns.add(nd);
		}
		samples = ns;

		String names[] = new String[] { "x", "y", "pop", "0to24", "25to64", "65older", "white", "black", "asian", "hispanic", "avgHHSize", "occup", "renterOccup" };

		DataUtils.writeShape(samples, geoms, names, "output/samples.shp");

		int[] fa = new int[samples.get(0).length - 2];
		for (int i = 0; i < fa.length; i++)
			fa[i] = i + 2;

		int[] ga = { 0, 1 };

		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa);

		DataUtils.zScoreColumns(samples, fa);
		// DataUtils.normalizeColumns(samples, fa);

		// DataUtils.writeToShape(samples, geoms, names,
		// "output/samples_zscore.shp");

		for (int l = 1; l <= N; l++) {

			log.debug(l);

			Sorter<double[]> bmuGetter = new KangasSorter<double[]>(gDist, fDist, l);
			NG ng = new NG(N, 10, 0.01, 0.5, 0.005, samples.get(0).length, bmuGetter);
			ng.initRandom(samples);

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

			// DataUtils.writeCSV(
			// "output/neurons_"+System.currentTimeMillis()+".csv",
			// ng.getNeurons(), new String[]{"x","y","v","c" } );

			// CHL
			Map<Connection, Integer> conns = new HashMap<Connection, Integer>();
			for (double[] x : samples) {
				bmuGetter.sort(x, ng.getNeurons());
				List<double[]> bmuList = ng.getNeurons();

				Connection c = new Connection(bmuList.get(0), bmuList.get(1));
				if (!conns.containsKey(c))
					conns.put(c, 1);
				else
					conns.put(c, conns.get(c) + 1);

				// test
				/*
				 * for( int i = 0; i < bmuList.size()-1; i++ ) { Connection c =
				 * new Connection( bmuList.get(i), bmuList.get(i+1) ); int
				 * invRank = bmuList.size()-i; if( !conns.containsKey(c) )
				 * conns.put( c, invRank); else conns.put( c, conns.get(c) +
				 * invRank ); }
				 */
			}

			Set<double[]> used = new HashSet<double[]>();
			for (Connection c : conns.keySet()) {
				used.add(c.getA());
				used.add(c.getB());
			}

			List<double[]> vertices = new ArrayList<double[]>(used);
			log.debug("usage: " + (double) vertices.size() / N);
			List<Connection> edges = new ArrayList<Connection>(conns.keySet());

			// DataUtils.writeCSV(
			// "output/vertices_"+System.currentTimeMillis()+".csv", vertices,
			// new String[]{"x","y","v","c" } );

			/*
			 * try { Drawer.geoDrawCluster(cluster.values(), samples, geoms, new
			 * FileOutputStream("output/clust_"+l+".png"), true); } catch
			 * (FileNotFoundException e2) { e2.printStackTrace(); }
			 */

			// write geoms+nodeids
			try {
				Writer w = new FileWriter("output/geoms_" + l + ".csv");
				w.write("id;geom\n");
				for (double[] d : vertices) {
					int id = vertices.indexOf(d);
					for (double[] x : cluster.get(d)) {
						int idx = samples.indexOf(x);
						Geometry g = geoms.get(idx);
						w.write(id + ";" + g + "\n");
					}
				}
				w.close();
			} catch (IOException e1) {
				e1.printStackTrace();
			}

			// write attributes
			try {
				Writer w = new FileWriter("output/attrs_" + l + ".csv");
				w.write("id;attr;value\n");
				for (double[] d : vertices) {
					int id = vertices.indexOf(d);
					for (double[] x : cluster.get(d)) {
						int idx = samples.indexOf(x);
						for (int i = 2; i < x.length; i++)
							// ignore first two attrs (x,y-coords)
							w.write(id + ";" + names[i] + ";" + x[i] + "\n");
					}
				}
				w.close();
			} catch (IOException e1) {
				e1.printStackTrace();
			}

			// write graphml
			Connection.writeGraphML(conns, "output/chl_" + l + ".graphml" );
		}
	}


}
