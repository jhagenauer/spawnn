package gwsom;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Point;

import nnet.SupervisedUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.dist.WeightedDist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.DefaultBmuGetter;
import spawnn.som.bmu.GWBmuGetter;
import spawnn.som.bmu.KangasBmuGetter;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.Transform;
import spawnn.utils.GeoUtils;
import spawnn.utils.GeoUtils.GWKernel;
import spawnn.utils.SpatialDataFrame;

public class GWSOM_Election {

	private static Logger log = Logger.getLogger(GWSOM_Election.class);

	public static void main(String[] args) {
		Random r = new Random();
		int T_MAX = 100000;

		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/election/election2004.shp"), true);
		for (int i = 0; i < sdf.samples.size(); i++) {
			Point p = sdf.geoms.get(i).getCentroid();
			sdf.samples.get(i)[0] = p.getX();
			sdf.samples.get(i)[1] = p.getY();
		}
		List<double[]> samples = sdf.samples;

		int[] ga = new int[] { 0, 1 };
		int[] fa = new int[] { 11, 21, 32, 54, 63, 48, 49 };
		
		for( int i : fa )
			log.debug(i+":"+sdf.names.get(i));

		Dist<double[]> fDist = new EuclideanDist(fa);
		Dist<double[]> gDist = new EuclideanDist(ga);

		DataUtils.transform(samples, fa, Transform.zScore);
		
		// --------------------
		
		/*{
		//samples = samples.subList(0, 10);
		Map<double[], Double> bandwidth = GeoUtils.getBandwidth(samples, gDist, 0.8, false);
		for( int i = 0; i < 3; i++ ) {
			double[] uv = samples.get(i);
			double[] g = GeoUtils.getGWMean( samples, uv, gDist, GWKernel.gaussian, bandwidth.get(uv) );
			log.debug("g: "+Arrays.toString( DataUtils.strip(g,fa)));
		}
		System.exit(1);
		}*/
		
		// --------------------
				
		boolean adaptive = false;
		
		List<Entry<List<Integer>, List<Integer>>> cvList = SupervisedUtils.getCVList(10, 1, samples.size());
		for (GWKernel ke : new GWKernel[] { GWKernel.gaussian })
			for (double bw = 0.1; bw <= 20; bw += 0.1 ) {

				List<double[]> rmse = new ArrayList<double[]>();
				for (final Entry<List<Integer>, List<Integer>> cvEntry : cvList) {
					List<double[]> samplesTrain = new ArrayList<double[]>();
					for (int k : cvEntry.getKey())
						samplesTrain.add(samples.get(k));

					List<double[]> samplesVal = new ArrayList<double[]>();
					for (int k : cvEntry.getValue())
						samplesVal.add(samples.get(k));

					Map<double[], Double> bandwidth = GeoUtils.getBandwidth(samplesVal, gDist, bw, adaptive);
					double[] e = new double[fa.length];
					for (double[] uv : samplesVal)
						for( int i = 0; i < fa.length; i++ )
							e[i] += Math.pow( uv[fa[i]] - GeoUtils.getGWMean(samplesTrain, uv, gDist, ke, bandwidth.get(uv))[fa[i]], 2);
					for( int i = 0; i < e.length; i++ )
						e[i] = Math.sqrt( e[i]/samplesVal.size() );
					
					rmse.add(e);
				}

				Map<double[], Double> bandwidth = GeoUtils.getBandwidth(samples, gDist, bw, adaptive);
				List<double[]> values = new ArrayList<>();
				for (double[] uv : samples) 
					values.add(GeoUtils.getGWMean(samples, uv, gDist, ke, bandwidth.get(uv)));
				
				//Drawer.geoDrawValues(sdf.geoms, values, fa[0], null, ColorBrewer.Blues, "output/gwmean_" + ke + "_" + bw + ".png");
				// DataUtils.writeCSV("output/gwmean_"+ke+"_"+bw+".csv", values,new String[]{"x","y","z"});

				double sum = 0;
				for( double[] e : rmse )
					for( double e2 : e )
						sum += e2;
				log.debug( ke + ", " + bw + "," + sum);
			}
		System.exit(1);
		

		Path file = Paths.get("output/results_election.csv");
		try {
			Files.createDirectories(file.getParent());
			Files.deleteIfExists(file);
			Files.createFile(file);
			String s = "method,parameter,variable,value\r\n";
			Files.write(file, s.getBytes(), StandardOpenOption.APPEND);
		} catch (IOException e1) {
			e1.printStackTrace();
		}
		
		log.debug("GWSOM");
		// for( GWKernel ke : GWKernel.values() )
		for (GWKernel ke : new GWKernel[] { GWKernel.gaussian })
			for (double bw = 0.01; bw <= 10; bw += 0.01 ) {
				
				Grid2DHex<double[]> grid = new Grid2DHex<double[]>(15, 20);
				SomUtils.initRandom(grid, samples);
				
				Map<double[], Double> bandwidth = GeoUtils.getBandwidth(samples, gDist, bw, adaptive);
				BmuGetter<double[]> bmuGetter = new GWBmuGetter(gDist, fDist, ke, bandwidth);

				SOM som = new SOM(new GaussKernel(new LinearDecay(10, 1)), new LinearDecay(1.0, 0.0), grid, bmuGetter);
				for (int t = 0; t < T_MAX; t++) {
					double[] x = samples.get(r.nextInt(samples.size()));
					som.train((double) t / T_MAX, x);
				}

				try {
					String s = ke+"," + bw + "," + SomUtils.getMeanQuantError(grid, bmuGetter, fDist, samples)
							+ SomUtils.getMeanQuantError(grid, bmuGetter, gDist, samples)
							+ SomUtils.getTopoError(grid, bmuGetter, samples) + "\r\n";
					Files.write(file, s.getBytes(), StandardOpenOption.APPEND);
				} catch (IOException e) {
					e.printStackTrace();
				}
			}

		log.debug("GeoSOM");
		for (int k = 0; k <= 26; k++ ) {
			Grid2DHex<double[]> grid = new Grid2DHex<double[]>(15, 20);
			log.debug(grid.getMaxDist());
			SomUtils.initRandom(grid, samples);
			BmuGetter<double[]> bmuGetter = new KangasBmuGetter<double[]>(gDist, fDist, k);

			SOM som = new SOM(new GaussKernel(new LinearDecay(10, 1)), new LinearDecay(1.0, 0.0), grid, bmuGetter);
			for (int t = 0; t < T_MAX; t++) {
				double[] x = samples.get(r.nextInt(samples.size()));
				som.train((double) t / T_MAX, x);
			}

			try {
				String s = "gwsom," + k + "," + SomUtils.getMeanQuantError(grid, bmuGetter, fDist, samples)
						+ SomUtils.getMeanQuantError(grid, bmuGetter, gDist, samples)
						+ SomUtils.getTopoError(grid, bmuGetter, samples) + "\r\n";
				Files.write(file, s.getBytes(), StandardOpenOption.APPEND);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		log.debug("WeightedSOM");
		for (double w = 0; w <= 1.0; w+=0.01 ) {
			Grid2DHex<double[]> grid = new Grid2DHex<double[]>(15, 20);
			SomUtils.initRandom(grid, samples);

			Map<Dist<double[]>, Double> m = new HashMap<>();
			m.put(gDist, w);
			m.put(fDist, 1.0 - w);
			Dist<double[]> wDist = new WeightedDist<>(m);
			BmuGetter<double[]> bmuGetter = new DefaultBmuGetter<>(wDist);

			SOM som = new SOM(new GaussKernel(new LinearDecay(10, 1)), new LinearDecay(1.0, 0.0), grid, bmuGetter);
			for (int t = 0; t < T_MAX; t++) {
				double[] x = samples.get(r.nextInt(samples.size()));
				som.train((double) t / T_MAX, x);
			}

			try {
				String s = "gwsom," + w + "," + SomUtils.getMeanQuantError(grid, bmuGetter, fDist, samples)
						+ SomUtils.getMeanQuantError(grid, bmuGetter, gDist, samples)
						+ SomUtils.getTopoError(grid, bmuGetter, samples) + "\r\n";
				Files.write(file, s.getBytes(), StandardOpenOption.APPEND);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
}
