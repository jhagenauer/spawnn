package gwsom;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Point;

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

		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/election/election2004.shp"),
				true);
		for (int i = 0; i < sdf.samples.size(); i++) {
			Point p = sdf.geoms.get(i).getCentroid();
			sdf.samples.get(i)[0] = p.getX();
			sdf.samples.get(i)[1] = p.getY();
		}
		List<double[]> samples = sdf.samples;

		int[] ga = new int[] { 0, 1 };
		int[] fa = new int[] { 11, 21, 23, 32, 54, 63, 48, 49 };

		Dist<double[]> fDist = new EuclideanDist(fa);
		Dist<double[]> gDist = new EuclideanDist(ga);

		DataUtils.transform(samples, fa, Transform.zScore);

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

		boolean adaptive = false;

		log.debug("GWSOM");
		// for( GWKernel ke : GWKernel.values() )
		for (GWKernel ke : new GWKernel[] { GWKernel.gaussian })
			for (double bw = 0; bw <= 10; bw += 0.01 ) {

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
					String s = "gwsom," + bw + "," + SomUtils.getMeanQuantError(grid, bmuGetter, fDist, samples)
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
