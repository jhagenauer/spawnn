import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Envelope;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryCollection;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.triangulate.VoronoiDiagramBuilder;

import cern.colt.Arrays;
import edu.uci.ics.jung.algorithms.shortestpath.DijkstraShortestPath;
import edu.uci.ics.jung.graph.DirectedSparseGraph;
import spawnn.utils.GraphUtils;

public class TestShortestPath {
	public static void main(String[] args) {
		GeometryFactory gf = new GeometryFactory();
		Random r = new Random();
		List<double[]> samples = new ArrayList<double[]>();
		List<Geometry> geoms = new ArrayList<Geometry>();
		List<Coordinate> coords = new ArrayList<Coordinate>();
		while (samples.size() < 20) {
			double x = r.nextDouble();
			double y = r.nextDouble();
			double z = r.nextDouble();
			Coordinate c = new Coordinate(x, y);
			coords.add(c);
			geoms.add(gf.createPoint(c));
			samples.add(new double[] { x, y, z });
		}

		VoronoiDiagramBuilder vdb = new VoronoiDiagramBuilder();
		vdb.setClipEnvelope(new Envelope(0, 0, 1, 1));
		vdb.setSites(coords);
		GeometryCollection coll = (GeometryCollection) vdb.getDiagram(gf);

		List<Geometry> voroGeoms = new ArrayList<Geometry>();
		for (int i = 0; i < coords.size(); i++) {
			Geometry p = gf.createPoint(coords.get(i));
			for (int j = 0; j < coll.getNumGeometries(); j++)
				if (p.intersects(coll.getGeometryN(j))) {
					voroGeoms.add(coll.getGeometryN(j));
					break;
				}
		}
		
		// build cm map based on voro
		Map<double[], Map<double[],Double>> cm = new HashMap<double[], Map<double[],Double>>();
		for (int i = 0; i < samples.size(); i++) {
			Map<double[],Double> s = new HashMap<double[],Double>();
			for (int j = 0; j < samples.size(); j++)
				if (i != j && voroGeoms.get(i).intersects(voroGeoms.get(j)))
					s.put(samples.get(j),voroGeoms.get(i).getCentroid().distance(voroGeoms.get(j).getCentroid()));
			cm.put(samples.get(i), s);
		}
		

		for( double[] a : cm.keySet() )
			for( Entry<double[],Double> e : cm.get(a).entrySet() )
				System.out.println(a+"-->"+e.getKey()+","+e.getValue() );

			
		double[] start = samples.get(r.nextInt(samples.size()));
		double[] end = samples.get(r.nextInt(samples.size()));
		System.out.println("start :"+start+", end: "+end);
		List<Entry<double[],Double>> p = GraphUtils.getDijkstraShortestPath(cm, start, end);
		System.out.println(p);
		
		// jung
		/*DirectedSparseGraph<double[], Double> graph = new DirectedSparseGraph<double[], Double>();
		for( double[] gp : cm.keySet()) {
			if (!graph.getVertices().contains(gp))
				graph.addVertex(gp);
			
			for( Entry<double[],Double> nb : cm.get(gp).entrySet() ) {
				if (!graph.getVertices().contains(nb.getKey()) )
					graph.addVertex(nb.getKey());
				System.out.println(Arrays.toString(gp)+","+Arrays.toString(nb.getKey())+","+nb.getValue());
				graph.addEdge(nb.getValue(), gp, nb.getKey());
			}
		}

		DijkstraShortestPath<double[], Double> dsp = new DijkstraShortestPath<double[], Double>(graph);
		System.out.println(dsp.getPath(start, end));*/
	}
}
