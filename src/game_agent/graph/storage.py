"""SQLite-based persistence for the UI state graph."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from game_agent.graph.models import EdgeAction, GraphEdge, GraphNode, UIStateGraph

logger = logging.getLogger(__name__)


class GraphStorage:
    """SQLite DAL for saving/loading the UI state machine graph."""

    def __init__(self, db_path: str = "data/graph.db") -> None:
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                page_name TEXT NOT NULL,
                page_description TEXT DEFAULT '',
                page_category TEXT DEFAULT '未知',
                key_buttons TEXT DEFAULT '[]',
                poco_tree_snapshot TEXT DEFAULT '',
                screenshot_path TEXT,
                discovered_at REAL DEFAULT 0,
                visit_count INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                discovered_at REAL DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                FOREIGN KEY (source_id) REFERENCES nodes(node_id),
                FOREIGN KEY (target_id) REFERENCES nodes(node_id)
            );

            CREATE TABLE IF NOT EXISTS edge_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                edge_id INTEGER NOT NULL,
                action_order INTEGER NOT NULL,
                action_type TEXT NOT NULL,
                target TEXT NOT NULL,
                params TEXT DEFAULT '{}',
                FOREIGN KEY (edge_id) REFERENCES edges(id)
            );
            """
        )

    def save_node(self, node: GraphNode) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO nodes
                (node_id, page_name, page_description, page_category,
                 key_buttons, poco_tree_snapshot, screenshot_path,
                 discovered_at, visit_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                node.node_id,
                node.page_name,
                node.page_description,
                node.page_category,
                json.dumps(node.key_buttons),
                node.poco_tree_snapshot,
                node.screenshot_path,
                node.discovered_at,
                node.visit_count,
            ),
        )
        self._conn.commit()

    def save_edge(self, edge: GraphEdge) -> int:
        cursor = self._conn.execute(
            """
            INSERT INTO edges (source_id, target_id, discovered_at, success_count, failure_count)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                edge.source_id,
                edge.target_id,
                edge.discovered_at,
                edge.success_count,
                edge.failure_count,
            ),
        )
        edge_id = cursor.lastrowid
        for i, action in enumerate(edge.actions):
            self._conn.execute(
                """
                INSERT INTO edge_actions (edge_id, action_order, action_type, target, params)
                VALUES (?, ?, ?, ?, ?)
                """,
                (edge_id, i, action.action_type, action.target, json.dumps(action.params)),
            )
        self._conn.commit()
        return edge_id

    def load_graph(self) -> UIStateGraph:
        graph = UIStateGraph()

        for row in self._conn.execute("SELECT * FROM nodes"):
            node = GraphNode(
                node_id=row["node_id"],
                page_name=row["page_name"],
                page_description=row["page_description"],
                page_category=row["page_category"],
                key_buttons=json.loads(row["key_buttons"]),
                poco_tree_snapshot=row["poco_tree_snapshot"],
                screenshot_path=row["screenshot_path"],
                discovered_at=row["discovered_at"],
                visit_count=row["visit_count"],
            )
            graph.add_node(node)

        for row in self._conn.execute("SELECT * FROM edges"):
            actions = []
            for action_row in self._conn.execute(
                "SELECT * FROM edge_actions WHERE edge_id = ? ORDER BY action_order",
                (row["id"],),
            ):
                actions.append(
                    EdgeAction(
                        action_type=action_row["action_type"],
                        target=action_row["target"],
                        params=json.loads(action_row["params"]),
                    )
                )
            edge = GraphEdge(
                source_id=row["source_id"],
                target_id=row["target_id"],
                actions=actions,
                discovered_at=row["discovered_at"],
                success_count=row["success_count"],
                failure_count=row["failure_count"],
            )
            graph.add_edge(edge)

        return graph

    def export_json(self, path: str) -> None:
        graph = self.load_graph()
        data = {
            "nodes": [
                {
                    "node_id": n.node_id,
                    "page_name": n.page_name,
                    "page_description": n.page_description,
                    "page_category": n.page_category,
                    "key_buttons": n.key_buttons,
                    "discovered_at": n.discovered_at,
                    "visit_count": n.visit_count,
                }
                for n in graph.all_nodes()
            ],
            "edges": [
                {
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "actions": [
                        {"action_type": a.action_type, "target": a.target, "params": a.params}
                        for a in e.actions
                    ],
                    "discovered_at": e.discovered_at,
                }
                for e in graph.all_edges()
            ],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def import_json(self, path: str) -> UIStateGraph:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        for node_data in data.get("nodes", []):
            node = GraphNode(**node_data)
            self.save_node(node)

        for edge_data in data.get("edges", []):
            actions = [EdgeAction(**a) for a in edge_data.pop("actions", [])]
            edge = GraphEdge(actions=actions, **edge_data)
            self.save_edge(edge)

        return self.load_graph()

    def close(self) -> None:
        self._conn.close()
