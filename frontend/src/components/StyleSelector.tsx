import type { StyleOption } from "../types";
import { API_BASE_URL } from "../api/client";

interface StyleSelectorProps {
  styles: StyleOption[];
  selectedStyleId: string | null;
  onSelect: (styleId: string) => void;
}

export function StyleSelector({
  styles,
  selectedStyleId,
  onSelect
}: StyleSelectorProps) {
  return (
    <div className="panel">
      <span className="label">Choose Style</span>
      <div className="style-grid">
        {styles.map((style) => (
          <button
            key={style.id}
            type="button"
            className={`style-card ${selectedStyleId === style.id ? "active" : ""}`}
            onClick={() => onSelect(style.id)}
          >
            <img src={new URL(style.preview, API_BASE_URL).toString()} alt={style.name} />
            <span>{style.name}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
