package clustering_cng;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.Reader;

import org.apache.commons.collections15.Transformer;

import spawnn.ng.NG;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;

import edu.uci.ics.jung.graph.UndirectedSparseGraph;
import edu.uci.ics.jung.io.GraphIOException;
import edu.uci.ics.jung.io.graphml.EdgeMetadata;
import edu.uci.ics.jung.io.graphml.GraphMLReader2;
import edu.uci.ics.jung.io.graphml.GraphMetadata;
import edu.uci.ics.jung.io.graphml.HyperEdgeMetadata;
import edu.uci.ics.jung.io.graphml.NodeMetadata;

public class BuildShapeFromGraph {

	public static void main(String[] args) {

		class Vertex {
			String id;
			double[] d;
			int color;
		}

		class Edge {
			String id;
			long weight;
		}

		UndirectedSparseGraph<Vertex, Edge> g;
		Reader reader;
		try {
			reader = new FileReader("output/best.graphml");

			Transformer<NodeMetadata, Vertex> vtrans = new Transformer<NodeMetadata, Vertex>() {
				public Vertex transform(NodeMetadata nmd) {
					Vertex v = new Vertex();
					v.d = new double[] { Double.valueOf(nmd.getProperty("v_n0")), Double.valueOf(nmd.getProperty("v_n1")), Double.valueOf(nmd.getProperty("v_n2")), Double.valueOf(nmd.getProperty("v_n3")), };
					v.color = Integer.valueOf(nmd.getProperty("v_color"));
					v.id = nmd.getId();
					return v;
				}
			};

			Transformer<EdgeMetadata, Edge> etrans = new Transformer<EdgeMetadata, Edge>() {
				public Edge transform(EdgeMetadata emd) {
					Edge e = new Edge();
					e.weight = Integer.valueOf(emd.getProperty("e_weight"));
					e.id = emd.getId();
					return e;
				}
			};

			Transformer<HyperEdgeMetadata, Edge> htrans = new Transformer<HyperEdgeMetadata, Edge>() {
				public Edge transform(HyperEdgeMetadata metadata) {
					Edge e = new Edge();
					return e;
				}
			};

			Transformer<GraphMetadata, UndirectedSparseGraph<Vertex, Edge>> gtrans = new Transformer<GraphMetadata, UndirectedSparseGraph<Vertex, Edge>>() {
				public UndirectedSparseGraph<Vertex, Edge> transform(GraphMetadata gmd) {
					return new UndirectedSparseGraph<Vertex, Edge>();
				}
			};

			GraphMLReader2<UndirectedSparseGraph<Vertex, Edge>, Vertex, Edge> gmlr = new GraphMLReader2<UndirectedSparseGraph<Vertex, Edge>, Vertex, Edge>(reader, gtrans, vtrans, etrans, htrans);
			g = gmlr.readGraph();


		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
		} catch (GraphIOException e) {
			e.printStackTrace();
		}
		
		// build cng
		
		// read shape

		// write shape

	}

}
