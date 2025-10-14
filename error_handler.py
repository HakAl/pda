def parse_api_error(e: Exception) -> str:
    """
    Parses a low-level API exception and returns a user-friendly string.

    Args:
        e: The exception caught during the API call.

    Returns:
        A clean, user-friendly error message.
    """
    error_str = str(e).lower()

    # Case 1: Rate Limit / Quota Exceeded (429)
    if "429" in error_str or "quota" in error_str or "resourceexhausted" in error_str:
        message = (
            "API rate limit exceeded. This often happens with free-tier keys "
            "that have strict usage limits.\n"
            "Please check your Google AI plan and billing details or wait a moment before trying again.\n"
            "For more info: https://ai.google.dev/gemini-api/docs/rate-limits"
        )
        return message

    # Case 2: Authentication Failed (401/403)
    if "401" in error_str or "403" in error_str or "permissiondenied" in error_str or "api key not valid" in error_str:
        return (
            "Authentication failed. Your API key may be invalid, expired, or missing "
            "the necessary permissions. Please validate your key and try again."
        )

    # Case 3: Model Not Found (404)
    if "404" in error_str and "model" in error_str:
        return (
            "The specified model was not found. It may be misspelled or you may not have "
            "access to it with your current API key."
        )

    # Case 4: Server-side Error (5xx)
    if "500" in error_str or "503" in error_str or "internal" in error_str or "unavailable" in error_str:
        return (
            "The AI service is currently unavailable or experiencing internal issues. "
            "Please try again in a few moments."
        )

    # Fallback for other RAGSystem or generic errors
    # Truncate long messages for better display
    original_error = str(e)
    if len(original_error) > 250:
        return f"An unexpected error occurred: {original_error[:250]}..."
    else:
        return f"An unexpected error occurred: {original_error}"