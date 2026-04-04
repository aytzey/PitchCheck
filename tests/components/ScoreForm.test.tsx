import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import ScoreForm from "@/components/ScoreForm";

describe("ScoreForm", () => {
  it("renders persona textarea, message textarea, and submit button", () => {
    render(<ScoreForm onScore={vi.fn()} loading={false} />);
    expect(screen.getByPlaceholderText(/startup background/i)).toBeDefined();
    expect(screen.getByPlaceholderText(/sales message/i)).toBeDefined();
    expect(screen.getByRole("button", { name: /score my pitch/i })).toBeDefined();
  });

  it("shows validation error for empty message", () => {
    const onScore = vi.fn();
    render(<ScoreForm onScore={onScore} loading={false} />);
    fireEvent.click(screen.getByRole("button", { name: /score my pitch/i }));
    expect(screen.getByRole("alert")).toBeDefined();
    expect(onScore).not.toHaveBeenCalled();
  });

  it("shows loading state", () => {
    render(<ScoreForm onScore={vi.fn()} loading={true} />);
    expect(screen.getByText(/scoring/i)).toBeDefined();
  });

  it("calls onScore with valid inputs", () => {
    const onScore = vi.fn();
    render(<ScoreForm onScore={onScore} loading={false} />);

    const personaInput = screen.getByPlaceholderText(/startup background/i);
    const messageInput = screen.getByPlaceholderText(/sales message/i);

    fireEvent.change(personaInput, { target: { value: "CTO at a startup, technical" } });
    fireEvent.change(messageInput, { target: { value: "Our platform reduces deployment time by 80% for enterprise teams" } });
    fireEvent.click(screen.getByRole("button", { name: /score my pitch/i }));

    expect(onScore).toHaveBeenCalledWith(
      "Our platform reduces deployment time by 80% for enterprise teams",
      "CTO at a startup, technical",
      "general"
    );
  });
});
