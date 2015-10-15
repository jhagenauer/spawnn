package cng_llm;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import llm.ErrorSorter;
import llm.LLMNG;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.ColorBrewerUtil.ColorMode;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.GeoUtils;
import spawnn.utils.SpatialDataFrame;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;

public class CreateSimulatedRefression {

	private static Logger log = Logger.getLogger(CreateSimulatedRefression.class);

	enum method {
		error, y, attr
	};

	public static void main(String[] args) {
		final Random r = new Random();
		
		final double spDep = 0.0;
		final double noise = 0.05;
		final List<double[]> samples = new ArrayList<double[]>();
		while (samples.size() < 2000) {
			double x = r.nextDouble();
			double lon = r.nextDouble();
			double lat = r.nextDouble();
			// double y = Math.pow(x,pow) * Math.ceil(lon*spDep)/spDep; //y depends on lon, relationship varies with lon

			/*
			 * if( r.nextDouble() <= spDep ) y *= lon; else y *= r.nextDouble();
			 */

			// y *= spDep*lon+(1.0-spDep)*r.nextDouble();
			// double y = x * (spDep*lon*lat+(1.0-spDep)*0.5) - noise/2 + r.nextDouble()*noise;
			double y = x * lon * lat /* + lat + lon */- noise / 2 + r.nextDouble() * noise;

			samples.add(new double[] { lat, lon, x, y });
		}

		DataUtils.writeCSV("output/simulatedRegression.csv", samples, new String[] { "lat", "lon", "x", "y" });
	}

}
