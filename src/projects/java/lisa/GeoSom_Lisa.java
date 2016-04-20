package lisa;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Geometry;

import cern.colt.Arrays;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.bmu.KangasBmuGetter;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.ColorBrewer;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.transform;
import spawnn.utils.Drawer;
import spawnn.utils.GeoUtils;
import spawnn.utils.SpatialDataFrame;

public class GeoSom_Lisa {

	private static Logger log = Logger.getLogger(GeoSom_Lisa.class);

	public static void main(String[] args) {

		final int T_MAX = 100000;
		final Random r = new Random();
		
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromCSV(new File("data/squareville.csv"), new int[]{0,1}, new int[]{}, true);
		final List<double[]> samples = sdf.samples;
		final List<Geometry> geoms = sdf.geoms;
		
		final int[] fa = new int[]{2};
		final int[] ga = new int[]{0,1};
		
		/*final SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/redcap/Election/election2004.shp"), true);
		final List<double[]> samples = sdf.samples;
		final List<Geometry> geoms = sdf.geoms;
		
		// add centroid
		for( int i = 0; i < samples.size(); i++ ) {
			Point p = geoms.get(i).getCentroid();
			double[] d = samples.get(i);
			d[2] = p.getX();
			d[3] = p.getY();
		}
		final int[] fa = new int[]{7}; // bush pct
		final int[] ga = new int[]{2,3};*/
		
		DataUtils.transform(samples, fa, transform.zScore);
		final Dist<double[]> fDist = new EuclideanDist(fa);
		final Dist<double[]> gDist = new EuclideanDist(ga);
		// ------------------------------------------------------------------------
		{
			
			
		Map<double[], Map<double[], Double>> dMap = GeoUtils.getRowNormedMatrix( GeoUtils.listsToWeights( GeoUtils.getKNNs(samples, gDist, 4, false) ) );
	
			
		//Map<double[], Map<double[], Double>> dMap = GeoUtils.getRowNormedMatrix(GeoUtils.listsToWeights(GeoUtils.getContiguityMap(samples, geoms, false, false)));

		Map<double[],Double> v = new HashMap<double[],Double>();
		for( double[] d : samples )
			v.put(d, d[fa[0]]);
		
		List<double[]> lisa = GeoUtils.getLocalMoransIMonteCarlo( samples, v, dMap, 10000);
		Drawer.geoDrawValues(geoms, lisa, 0, sdf.crs, ColorBrewer.Blues, "output/lisa_mc.png");

		List<Double> signf = new ArrayList<Double>();
		for (double[] d : lisa)
			if (d[4] < 0.0001)
				signf.add(0.0);
			else if (d[4] < 0.001)
				signf.add(1.0);
			else if (d[4] < 0.01)
				signf.add(2.0);
			else if (d[4] < 0.05)
				signf.add(3.0);
			else
				signf.add(4.0);
		Drawer.geoDrawValues(geoms, signf, sdf.crs, ColorBrewer.Spectral, "output/lisa_mc_signf.png");

		DescriptiveStatistics ds = new DescriptiveStatistics();
		for (double d : v.values() )
			ds.addValue(d);
		double mean = ds.getMean();

		final Map<Integer, Set<double[]>> lisaCluster = new HashMap<Integer, Set<double[]>>();
		for (int i = 0; i < samples.size(); i++) {
			double[] l = lisa.get(i);
			double d = v.get(samples.get(i));
			int clust = -1;

			if (l[4] > 0.05) // not significant
				clust = 0;
			else if (l[0] > 0 && d > mean)
				clust = 1; // high-high
			else if (l[0] > 0 && d < mean)
				clust = 2; // low-low
			else if (l[0] < 0 && d > mean)
				clust = 3; // high-low
			else if (l[0] < 0 && d < mean)
				clust = 4; // low-high
			else
				clust = 5; // unknown

			if (!lisaCluster.containsKey(clust))
				lisaCluster.put(clust, new HashSet<double[]>());
			lisaCluster.get(clust).add(samples.get(i));
		}
		Drawer.geoDrawCluster(lisaCluster.values(), samples, geoms, "output/lisa_mc_clust.png", true);

		log.debug("Moran's I: "+ Arrays.toString( GeoUtils.getMoransIStatistics(dMap, v ) ) );
		log.debug("Moran's I mc: "+ Arrays.toString( GeoUtils.getMoransIStatisticsMonteCarlo(dMap, v, 10000) ) );
		
		double s = 0;
		for( double[] d : lisa )
			s += d[0];
		log.debug( (s/lisa.size()) );
		}
		// -----------------------------------------------------------------------------------------
		for( int k = 1; k <= 7; k++ ) {
			Grid2D<double[]> grid = new Grid2DHex<double[]>(12, 8);
			SomUtils.initRandom(grid, samples);
			
			KangasBmuGetter<double[]> bg = new KangasBmuGetter<double[]>(gDist, fDist, k);
			SOM som = new SOM(new GaussKernel(new LinearDecay(grid.getMaxDist(), 1)), new LinearDecay(1.0, 0.005), grid, bg);
			for (int t = 0; t < T_MAX; t++) {
				double[] x = samples.get(r.nextInt(samples.size()));
				som.train((double) t / T_MAX, x);
			}
			
			SomUtils.printDMatrix(grid, fDist, "output/"+k+"_dmatrix.png");
			
			Map<double[], Map<double[], Double>> dMap = GeoUtils.getRowNormedMatrix(GeoUtils.listsToWeights(grid.getContiguityMap()));
			List<double[]> pts = new ArrayList<>(dMap.keySet());
			
			Map<double[],Double> v = new HashMap<double[],Double>();
			for( double[] d : pts )
				v.put(d, d[fa[0]]);
			
			List<double[]> lisa = GeoUtils.getLocalMoransIMonteCarlo( pts, v, dMap, 10000);
						
			// draw lisa-values
			Grid2DHex<double[]> g = new Grid2DHex<>(grid.getSizeOfDim(0),grid.getSizeOfDim(1));
			for( GridPos p : grid.getPositions() ) 
				g.setPrototypeAt(p, lisa.get(pts.indexOf(grid.getPrototypeAt(p))) );
			try {
				SomUtils.printImage( SomUtils.getHexMatrixImage(g, 10, ColorBrewer.Blues, SomUtils.HEX_NORMAL, 0), new FileOutputStream("output/"+k+"_lisa_mc.png") );
			} catch (FileNotFoundException e1) {
				e1.printStackTrace();
			}
			
			// draw lisa-signf
			List<Double> signf = new ArrayList<Double>();
			for (double[] d : lisa)
				if (d[4] < 0.0001)
					signf.add(0.0);
				else if (d[4] < 0.001)
					signf.add(1.0);
				else if (d[4] < 0.01)
					signf.add(2.0);
				else if (d[4] < 0.05)
					signf.add(3.0);
				else
					signf.add(4.0);
			
			for( GridPos p : grid.getPositions() ) 
				g.setPrototypeAt(p, new double[]{signf.get(pts.indexOf(grid.getPrototypeAt(p)))} );
			try {
				SomUtils.printImage( SomUtils.getHexMatrixImage(g, 10, ColorBrewer.Spectral, SomUtils.HEX_NORMAL, 0), new FileOutputStream("output/"+k+"_lisa_mc_signf.png") );
			} catch (FileNotFoundException e1) {
				e1.printStackTrace();
			}
			
			// draw lisa-cluster
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for (double d : v.values() )
				ds.addValue(d);
			double mean = ds.getMean();

			List<Double> clust = new ArrayList<Double>();
			for (int i = 0; i < pts.size(); i++) {
				double[] l = lisa.get(i);
				double d = v.get(pts.get(i));
				int c = -1;
				if (l[4] > 0.05) // not significant
					c = 0;
				else if (l[0] > 0 && d > mean)
					c = 1; // high-high
				else if (l[0] > 0 && d < mean)
					c = 2; // low-low
				else if (l[0] < 0 && d > mean)
					c = 3; // high-low
				else if (l[0] < 0 && d < mean)
					c = 4; // low-high
				else
					c = 5; // unknown
				clust.add((double)c);
			}
			
			for( GridPos p : grid.getPositions() ) 
				g.setPrototypeAt(p, new double[]{clust.get(pts.indexOf(grid.getPrototypeAt(p)))} );
			try {
				SomUtils.printImage( SomUtils.getHexMatrixImage(g, 10, ColorBrewer.Spectral, SomUtils.HEX_NORMAL, 0), new FileOutputStream("output/"+k+"_lisa_mc_clust.png") );
			} catch (FileNotFoundException e1) {
				e1.printStackTrace();
			}
			
			log.debug(k+":"+Arrays.toString(GeoUtils.getMoransIStatistics(dMap, v)));
		}
	}
}
