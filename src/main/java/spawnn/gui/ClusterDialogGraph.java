package spawnn.gui;

import java.awt.CardLayout;
import java.awt.Frame;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.DefaultComboBoxModel;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JDialog;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;

import net.miginfocom.swing.MigLayout;

public class ClusterDialogGraph extends JDialog implements ActionListener {

	JPanel cards;
	JComboBox cb, weights;

	JButton ok, cancel;
	boolean okPressed = false;

	JTextField numEdgesToRemove, numCandidates, maxCluster, restarts;

	public static enum ClusterAlgorithm {MultilevelModularity, EdgeBetweeness, Voltage };

	private static final long serialVersionUID = -6577606485481195079L;

	public ClusterDialogGraph(Frame parent, String string, boolean b) {
		super(parent, string, b);

		setLayout(new MigLayout(""));

		add(new JLabel("Algorithm:"));
		cb = new JComboBox();
		cb.setModel(new DefaultComboBoxModel(ClusterAlgorithm.values()));
		cb.removeItem(ClusterAlgorithm.Voltage); // The implementation seems buggy! clusters are not always connected

		cb.addActionListener(this);
		add(cb, "wrap");

		JPanel eb = new JPanel(new MigLayout(""));
		eb.add(new JLabel("Number of edges to remove"));
		numEdgesToRemove = new JTextField("5", 3);
		eb.add(numEdgesToRemove);

		JPanel volt = new JPanel(new MigLayout(""));
		volt.add(new JLabel("Number of candidates:"));
		numCandidates = new JTextField("5", 3);
		volt.add(numCandidates, "wrap");

		maxCluster = new JTextField("5", 3);
		volt.add(new JLabel("Max. number of clusters:"));
		volt.add(maxCluster, "wrap");
		
		JPanel greedy = new JPanel(new MigLayout(""));
		greedy.add(new JLabel("Weights:"));
		weights = new JComboBox();
		weights.addItem(GraphPanel.NONE);
		weights.addItem(GraphPanel.COUNT);
		weights.addItem(GraphPanel.DIST);
		weights.addItem(GraphPanel.DIST_GEO);
		greedy.add(weights,"wrap");
		greedy.add(new JLabel("Number of random restarts:") );
		restarts = new JTextField("10",3);
		greedy.add(restarts, "wrap");
		
		cards = new JPanel(new CardLayout());
		cards.add(greedy, ClusterAlgorithm.MultilevelModularity.toString());
		cards.add(eb, ClusterAlgorithm.EdgeBetweeness.toString());
		cards.add(volt, ClusterAlgorithm.Voltage.toString()); 
				
		add(cards, "span 2, wrap");

		ok = new JButton("OK");
		ok.addActionListener(this);
		cancel = new JButton("Cancel");
		cancel.addActionListener(this);
		add(ok, "split 2, push");
		add(cancel);

		setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		pack();
		setAlwaysOnTop(true);
		setLocationRelativeTo(parent);
		setVisible(true);
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == cb) {
			CardLayout cl = (CardLayout) (cards.getLayout());
			cl.show(cards, cb.getSelectedItem().toString());
		} else if (e.getSource() == ok) {
			okPressed = true;
			dispose();
		} else {
			okPressed = false;
			dispose();
		}
	}

	public boolean isOkPressed() {
		return okPressed;
	}

	public ClusterAlgorithm getAlgorithm() {
		return (ClusterAlgorithm) cb.getSelectedItem();
	}

	public int getNumEdgesToRemove() {
		return Integer.parseInt(numEdgesToRemove.getText());
	}

	public int getNumCandidates() {
		return Integer.parseInt(numCandidates.getText());
	}

	public int getMaxCluster() {
		return Integer.parseInt(maxCluster.getText());
	}
	
	public int getRestarts() {
		return Integer.parseInt(restarts.getText());
	}
	
	public String getWeights() {
		return weights.getSelectedItem().toString();
	}
}
