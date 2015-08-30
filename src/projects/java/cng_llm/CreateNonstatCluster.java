package cng_llm;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.log4j.Logger;

import spawnn.utils.DataUtils;
import spawnn.utils.SampleBuilder;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Envelope;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryCollection;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.Point;
import com.vividsolutions.jts.triangulate.VoronoiDiagramBuilder;

public class CreateNonstatCluster {
	
	private static Logger log = Logger.getLogger(SampleBuilder.class);

	public static void main(String[] args) {
		buildNonstationaryCluster(25, 1000, "output/cluster.shp");
	}

	public static void buildNonstationaryCluster(int numCluster, int numSamples, String fn) {
		int maxColors = 4;
		final Random r = new Random();

		Envelope env = new Envelope(0, 1, 0, 1);
		List<Coordinate> coords = new ArrayList<Coordinate>();
		for (int i = 0; i < numSamples; i++)
			coords.add(new Coordinate(r.nextDouble(), r.nextDouble()));
		VoronoiDiagramBuilder vdb = new VoronoiDiagramBuilder();
		vdb.setSites(coords);

		GeometryFactory gf = new GeometryFactory();
		GeometryCollection coll = (GeometryCollection) vdb.getDiagram(gf);

		List<Geometry> geoms = new ArrayList<Geometry>();
		List<double[]> points = new ArrayList<double[]>();

		for (int i = 0; i < coll.getNumGeometries(); i++) {
			Geometry geom = coll.getGeometryN(i);
			geom = geom.intersection(gf.toGeometry(env));

			Point p = geom.getCentroid();

			geoms.add(geom);
			double[] d = new double[] { p.getX(), p.getY() }; 
			points.add(d);
		}

		Map<Integer, Set<double[]>> clusterMap = new HashMap<Integer, Set<double[]>>();
		for (int i = 0; i < numCluster; i++)
			clusterMap.put(i, new HashSet<double[]>());

		// seed
		Set<double[]> seed = new HashSet<double[]>();
		for( int i : clusterMap.keySet() ) {
			double[] d = null;
			do {
				d = points.get(r.nextInt(points.size()));
			} while( seed.contains(d) );
			clusterMap.get(i).add(d);
			seed.add(d);
		}
		
		// grow cluster as long unassigned samples exist
		Set<double[]> assigned = new HashSet<double[]>(seed);
		while (assigned.size() < points.size() ) {
			// randomly get a cluster and sample
			List<Integer> keys = new ArrayList<Integer>(clusterMap.keySet());
			int c = keys.get(r.nextInt(keys.size()));
			
			List<double[]> l = new ArrayList<double[]>(clusterMap.get(c));
			double[] d = l.get(r.nextInt(l.size()));
			Geometry geom = geoms.get(points.indexOf(d));

			// check surrounding samples for free one
			List<double[]> candidates = new ArrayList<double[]>();
			for (int i = 0; i < points.size(); i++) {
				if (geoms.get(i).touches(geom) && !assigned.contains(points.get(i) ) )
					candidates.add(points.get(i));
			}

			if (candidates.size() > 0) {
				double[] d2 = candidates.get(r.nextInt(candidates.size()));
				clusterMap.get(c).add(d2);
				assigned.add(d2);
			}
		}

		int curNrColors = 0;
		Map<double[],Integer> colorMap = new HashMap<double[],Integer>();
		do {
			log.debug("greedy coloring");
			for( double[] d : points )
				colorMap.put(d, 0);
			
			List<Integer> cs = new ArrayList<Integer>(clusterMap.keySet());
			Collections.shuffle(cs);

			for (int c : cs) {
				Set<double[]> l = clusterMap.get(c);

				// get all neighbors of l
				List<double[]> nbs = new ArrayList<double[]>();
				for (double[] d : l) {
					int idx = points.indexOf(d);
					for (int j = 0; j < points.size(); j++) {
						if (geoms.get(j).touches(geoms.get(idx)) && !l.contains(points.get(j)))
							nbs.add(points.get(j));
					}
				}

				// get neighboring colors
				Set<Integer> nbC = new HashSet<Integer>();
				for (double[] d : nbs)
					for (int nc : clusterMap.keySet()) 
						if (clusterMap.get(nc).contains(d))
							nbC.add( colorMap.get(d) );
				
				// use lowest not used color
				int color = 0;
				for (; nbC.contains(color); color++)
					;

				// assign color
				for (double[] d : l)
					colorMap.put(d, color);
			}

			curNrColors = new HashSet<Integer>(colorMap.values()).size();
			log.debug("Colors: " + curNrColors);
		} while (curNrColors > maxColors);
		
		// add class value
		List<double[]> samples = new ArrayList<double[]>();
		for (double[] d : points) {
			
			int c = colorMap.get(d); // only few colors
			int cl = -1; // many cluster
			for ( int i : clusterMap.keySet() )
				if (clusterMap.get(i).contains(d))
					cl = i;
			
			double x = r.nextDouble();
			double y = 0;
			if( c == 0 ) {
				y = x + 1;
			} else if( c == 1 ) {
				y = x - 1;
			} else if( c == 2 ) {
				y = -x + 1;
			} else if( c == 3 ) {
				y = -y - 1;
			}
			
			double[] nd = new double[]{ d[0], d[1], x, y, c };
			samples.add(nd);
		}

		DataUtils.writeShape(samples, geoms, new String[] { "lat", "lon", "x", "y", "class" }, fn);
	}
}
